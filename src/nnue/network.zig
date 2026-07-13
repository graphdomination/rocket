const std = @import("std");
const GameState = @import("../gamestate.zig").GameState;
const Move = @import("../movegen.zig").Move;
const features = @import("features.zig");
const accumulator = @import("accumulator.zig");
const Accumulator = accumulator.Accumulator;
const AccumulatorStack = accumulator.AccumulatorStack;
const AccumulatorUpdater = accumulator.AccumulatorUpdater;

/// Network architecture constants (Koivisto format)
pub const INPUT_SIZE: usize = features.INPUT_SIZE; // 12,288
pub const HIDDEN_SIZE: usize = accumulator.HIDDEN_SIZE; // 512
pub const HIDDEN_DSIZE: usize = HIDDEN_SIZE * 2; // 1024 (both perspectives)
pub const OUTPUT_SIZE: usize = 1;
const SIMD_LANES: usize = 16;
const VecI16 = @Vector(SIMD_LANES, i16);
const VecI32 = @Vector(SIMD_LANES, i32);

/// Quantization multipliers
pub const INPUT_WEIGHT_MULTIPLIER: i32 = 32;
pub const HIDDEN_WEIGHT_MULTIPLIER: i32 = 128;

/// NNUE Network structure
pub const Network = struct {
    // Network weights and biases
    input_weights: [][HIDDEN_SIZE]i16, // [INPUT_SIZE][HIDDEN_SIZE]
    input_biases: []i16, // [HIDDEN_SIZE]
    hidden_weights: [][HIDDEN_DSIZE]i16, // [OUTPUT_SIZE][HIDDEN_DSIZE]
    hidden_biases: []i32, // [OUTPUT_SIZE]

    // Accumulator management
    acc_stack: AccumulatorStack,
    updater: AccumulatorUpdater,

    allocator: std.mem.Allocator,
    loaded: bool,
    owns_weights: bool,

    pub fn init(allocator: std.mem.Allocator) !Network {
        // Allocate network weights
        const input_weights = try allocator.alloc([HIDDEN_SIZE]i16, INPUT_SIZE);
        const input_biases = try allocator.alloc(i16, HIDDEN_SIZE);
        const hidden_weights = try allocator.alloc([HIDDEN_DSIZE]i16, OUTPUT_SIZE);
        const hidden_biases = try allocator.alloc(i32, OUTPUT_SIZE);

        // Initialize to zero
        @memset(input_weights, [_]i16{0} ** HIDDEN_SIZE);
        @memset(input_biases, 0);
        @memset(hidden_weights, [_]i16{0} ** HIDDEN_DSIZE);
        @memset(hidden_biases, 0);

        // Create accumulator stack (max depth 128)
        const acc_stack = try AccumulatorStack.init(allocator, 128);

        return Network{
            .input_weights = input_weights,
            .input_biases = input_biases,
            .hidden_weights = hidden_weights,
            .hidden_biases = hidden_biases,
            .acc_stack = acc_stack,
            .updater = AccumulatorUpdater.init(input_weights, input_biases),
            .allocator = allocator,
            .loaded = false,
            .owns_weights = true,
        };
    }

    /// Create a per-search context which shares immutable weights but owns its
    /// accumulator stack. A Network itself is not thread-safe because search
    /// continuously pushes and updates that stack.
    pub fn initShared(allocator: std.mem.Allocator, source: *const Network) !Network {
        const acc_stack = try AccumulatorStack.init(allocator, 128);
        return Network{
            .input_weights = source.input_weights,
            .input_biases = source.input_biases,
            .hidden_weights = source.hidden_weights,
            .hidden_biases = source.hidden_biases,
            .acc_stack = acc_stack,
            .updater = AccumulatorUpdater.init(source.input_weights, source.input_biases),
            .allocator = allocator,
            .loaded = source.loaded,
            .owns_weights = false,
        };
    }

    pub fn deinit(self: *Network) void {
        if (self.owns_weights) {
            self.allocator.free(self.input_weights);
            self.allocator.free(self.input_biases);
            self.allocator.free(self.hidden_weights);
            self.allocator.free(self.hidden_biases);
        }
        self.acc_stack.deinit();
    }

    /// Check if network is loaded and ready
    pub fn isLoaded(self: *const Network) bool {
        return self.loaded;
    }

    /// Initialize accumulator for a new position
    pub fn initPosition(self: *Network, state: *const GameState) void {
        self.acc_stack.reset();
        self.updater.refresh(self.acc_stack.current(), state);
        self.acc_stack.needs_refresh = false;
    }

    /// Push accumulator before making a move
    pub fn pushAccumulator(self: *Network) void {
        self.acc_stack.push();
    }

    /// Pop accumulator after unmaking a move
    pub fn popAccumulator(self: *Network) void {
        self.acc_stack.pop();
    }

    /// Push and construct a child accumulator without first copying 2 KB from
    /// the parent. Ordinary moves use one fused delta pass; special moves fall
    /// back to a complete refresh of the new frame.
    pub fn pushAndUpdateForMove(
        self: *Network,
        state: *const GameState,
        move: Move,
        captured_piece: ?@import("../piece.zig").PieceKind,
    ) void {
        const parent = self.acc_stack.currentConst();
        self.acc_stack.pushEmpty();
        const child = self.acc_stack.current();
        if (!self.updater.updateMoveFromParent(parent, child, state, move, captured_piece)) {
            self.updater.refresh(child, state);
        }
        self.acc_stack.needs_refresh = false;
    }

    /// Update accumulator for a move (incremental or refresh)
    pub fn updateForMove(
        self: *Network,
        state: *const GameState,
        move: Move,
        captured_piece: ?@import("../piece.zig").PieceKind,
    ) void {
        const acc = self.acc_stack.current();

        // Try incremental update
        const updated = self.updater.updateMove(acc, state, move, captured_piece);

        // Refresh this frame immediately. A single dirty flag shared by the
        // whole stack is insufficient: a deeper child could clear it and then
        // expose a stale parent after pop.
        if (!updated) {
            self.updater.refresh(acc, state);
            self.acc_stack.needs_refresh = false;
        }
    }

    /// Evaluate position using NNUE
    /// Returns evaluation in centipawns from perspective of side to move
    pub fn evaluate(self: *Network, state: *const GameState) i32 {
        if (!self.loaded) return 0;

        // Refresh accumulator if needed
        if (self.acc_stack.needs_refresh) {
            self.updater.refresh(self.acc_stack.current(), state);
            self.acc_stack.needs_refresh = false;
        }

        const acc = self.acc_stack.currentConst();

        // Get active side (side to move)
        const active_perspective = state.side_to_move; // 1 = white, 0 = black

        const acc_active = if (active_perspective == 1) &acc.white else &acc.black;
        const acc_non_active = if (active_perspective == 1) &acc.black else &acc.white;

        // Forward pass: accumulator -> hidden layer -> output
        return self.forward(acc_active, acc_non_active);
    }

    /// Forward pass through the network
    fn forward(self: *const Network, acc_active: []const i16, acc_non_active: []const i16) i32 {
        var result: i32 = self.hidden_biases[0];
        const zero: VecI16 = @splat(0);

        var i: usize = 0;
        while (i < HIDDEN_SIZE) : (i += SIMD_LANES) {
            const active: VecI16 = @bitCast(acc_active[i..][0..SIMD_LANES].*);
            const passive: VecI16 = @bitCast(acc_non_active[i..][0..SIMD_LANES].*);
            const active_weights: VecI16 = @bitCast(self.hidden_weights[0][i..][0..SIMD_LANES].*);
            const passive_weights: VecI16 = @bitCast(self.hidden_weights[0][HIDDEN_SIZE + i ..][0..SIMD_LANES].*);

            // Koivisto's network uses an ordinary ReLU here. It deliberately
            // does not clip positive accumulator values to 127.
            const active_relu = @max(active, zero);
            const passive_relu = @max(passive, zero);
            const active_product: VecI32 = @as(VecI32, @intCast(active_relu)) * @as(VecI32, @intCast(active_weights));
            const passive_product: VecI32 = @as(VecI32, @intCast(passive_relu)) * @as(VecI32, @intCast(passive_weights));
            result += @reduce(.Add, active_product + passive_product);
        }

        // Dequantize: divide by both multipliers
        return @divTrunc(result, INPUT_WEIGHT_MULTIPLIER * HIDDEN_WEIGHT_MULTIPLIER);
    }

};

/// Helper to get evaluation with proper error handling
pub fn evaluatePosition(network: *Network, state: *const GameState) i32 {
    if (!network.loaded) {
        return 0; // Fallback to classical eval
    }
    return network.evaluate(state);
}
