const std = @import("std");
const GameState = @import("../gamestate.zig").GameState;
const Move = @import("../movegen.zig").Move;
const PieceKind = @import("../piece.zig").PieceKind;
const features = @import("features.zig");

pub const HIDDEN_SIZE: usize = 512;
const SIMD_LANES: usize = 16;
const VecI16 = @Vector(SIMD_LANES, i16);

/// Accumulator stores the partially computed first layer activations
/// One accumulator per perspective (white and black)
pub const Accumulator = struct {
    white: [HIDDEN_SIZE]i16, // White's perspective
    black: [HIDDEN_SIZE]i16, // Black's perspective

    pub fn init() Accumulator {
        return Accumulator{
            .white = [_]i16{0} ** HIDDEN_SIZE,
            .black = [_]i16{0} ** HIDDEN_SIZE,
        };
    }

    /// Copy from another accumulator
    pub fn copyFrom(self: *Accumulator, other: *const Accumulator) void {
        @memcpy(&self.white, &other.white);
        @memcpy(&self.black, &other.black);
    }

    /// Reset to biases
    pub fn reset(self: *Accumulator, biases: []const i16) void {
        @memcpy(&self.white, biases);
        @memcpy(&self.black, biases);
    }
};

/// Accumulator stack for efficient updates during search
pub const AccumulatorStack = struct {
    stack: []Accumulator,
    index: usize,
    allocator: std.mem.Allocator,
    needs_refresh: bool,

    // Cache for tracking king positions
    last_white_king: ?u8,
    last_black_king: ?u8,

    pub fn init(allocator: std.mem.Allocator, max_depth: usize) !AccumulatorStack {
        const stack = try allocator.alloc(Accumulator, max_depth);
        for (stack) |*acc| {
            acc.* = Accumulator.init();
        }

        return AccumulatorStack{
            .stack = stack,
            .index = 0,
            .allocator = allocator,
            .needs_refresh = true,
            .last_white_king = null,
            .last_black_king = null,
        };
    }

    pub fn deinit(self: *AccumulatorStack) void {
        self.allocator.free(self.stack);
    }

    /// Get current accumulator
    pub fn current(self: *AccumulatorStack) *Accumulator {
        return &self.stack[self.index];
    }

    /// Get current accumulator (const)
    pub fn currentConst(self: *const AccumulatorStack) *const Accumulator {
        return &self.stack[self.index];
    }

    /// Push a new accumulator level (copy from current)
    pub fn push(self: *AccumulatorStack) void {
        if (self.index + 1 < self.stack.len) {
            self.index += 1;
            self.stack[self.index].copyFrom(&self.stack[self.index - 1]);
        }
    }

    /// Advance to a child frame without copying. The caller must overwrite the
    /// complete accumulator before it is evaluated.
    pub fn pushEmpty(self: *AccumulatorStack) void {
        if (self.index + 1 < self.stack.len) self.index += 1;
    }

    /// Pop accumulator level
    pub fn pop(self: *AccumulatorStack) void {
        if (self.index > 0) {
            self.index -= 1;
        }
    }

    /// Reset to root
    pub fn reset(self: *AccumulatorStack) void {
        self.index = 0;
        self.needs_refresh = true;
        self.last_white_king = null;
        self.last_black_king = null;
    }

    /// Mark as needing refresh
    pub fn invalidate(self: *AccumulatorStack) void {
        self.needs_refresh = true;
    }
};

/// Accumulator updater - handles incremental updates
pub const AccumulatorUpdater = struct {
    input_weights: []const [HIDDEN_SIZE]i16,
    input_biases: []const i16,

    pub fn init(input_weights: []const [HIDDEN_SIZE]i16, input_biases: []const i16) AccumulatorUpdater {
        return AccumulatorUpdater{
            .input_weights = input_weights,
            .input_biases = input_biases,
        };
    }

    /// Refresh accumulator from scratch
    pub fn refresh(self: *const AccumulatorUpdater, acc: *Accumulator, state: *const GameState) void {
        // Reset to biases
        acc.reset(self.input_biases);

        // GameState maintains these incrementally in make/unmakeMove. Reading
        // them directly avoids two full-board scans on every NNUE update.
        const white_king = state.king_squares[0];
        const black_king = state.king_squares[1];

        // Add all pieces
        for (state.board.squares, 0..) |square, idx| {
            if (square.piece) |piece| {
                const feature = features.getFeatureWithKings(
                    piece.kind,
                    piece.color,
                    @intCast(idx),
                    white_king,
                    black_king,
                );
                self.addFeature(acc, feature);
            }
        }
    }

    /// Add a feature to the accumulator
    pub fn addFeature(self: *const AccumulatorUpdater, acc: *Accumulator, feature: features.Feature) void {
        const white_idx = feature.index_white;
        const black_idx = feature.index_black;

        if (white_idx >= self.input_weights.len or black_idx >= self.input_weights.len) {
            return;
        }

        var i: usize = 0;
        while (i < HIDDEN_SIZE) : (i += SIMD_LANES) {
            const acc_white: VecI16 = @bitCast(acc.white[i..][0..SIMD_LANES].*);
            const acc_black: VecI16 = @bitCast(acc.black[i..][0..SIMD_LANES].*);
            const weights_white: VecI16 = @bitCast(self.input_weights[white_idx][i..][0..SIMD_LANES].*);
            const weights_black: VecI16 = @bitCast(self.input_weights[black_idx][i..][0..SIMD_LANES].*);
            acc.white[i..][0..SIMD_LANES].* = @bitCast(acc_white +% weights_white);
            acc.black[i..][0..SIMD_LANES].* = @bitCast(acc_black +% weights_black);
        }
    }

    /// Remove a feature from the accumulator
    pub fn removeFeature(self: *const AccumulatorUpdater, acc: *Accumulator, feature: features.Feature) void {
        const white_idx = feature.index_white;
        const black_idx = feature.index_black;

        if (white_idx >= self.input_weights.len or black_idx >= self.input_weights.len) {
            return;
        }

        var i: usize = 0;
        while (i < HIDDEN_SIZE) : (i += SIMD_LANES) {
            const acc_white: VecI16 = @bitCast(acc.white[i..][0..SIMD_LANES].*);
            const acc_black: VecI16 = @bitCast(acc.black[i..][0..SIMD_LANES].*);
            const weights_white: VecI16 = @bitCast(self.input_weights[white_idx][i..][0..SIMD_LANES].*);
            const weights_black: VecI16 = @bitCast(self.input_weights[black_idx][i..][0..SIMD_LANES].*);
            acc.white[i..][0..SIMD_LANES].* = @bitCast(acc_white -% weights_white);
            acc.black[i..][0..SIMD_LANES].* = @bitCast(acc_black -% weights_black);
        }
    }

    /// Update accumulator for a move (incremental)
    /// Returns true if incremental update was done, false if refresh is needed
    pub fn updateMove(
        self: *const AccumulatorUpdater,
        acc: *Accumulator,
        state: *const GameState,
        move: Move,
        captured_piece: ?PieceKind,
    ) bool {
        const white_king = state.king_squares[0];
        const black_king = state.king_squares[1];

        // Get moving piece (it's now at 'to' square after move is applied)
        const moving_piece = state.board.squares[move.to].piece orelse return false;

        // For king moves, always refresh (king bucket affects all pieces)
        if (moving_piece.kind == .King) {
            return false;
        }

        // Castling, en passant, and promotion move more than one ordinary
        // feature (or change the moving piece type), so refresh them safely.
        if (move.is_castle or move.is_en_passant or move.promotion != null) {
            return false;
        }

        // Standard incremental update

        // Remove piece from old square
        const old_feature = features.getFeatureWithKings(
            moving_piece.kind,
            moving_piece.color,
            move.from,
            white_king,
            black_king,
        );
        self.removeFeature(acc, old_feature);

        // Handle capture (remove captured piece)
        if (captured_piece) |cap_kind| {
            const cap_color = 1 - moving_piece.color; // Opposite color
            const cap_feature = features.getFeatureWithKings(
                cap_kind,
                cap_color,
                move.to,
                white_king,
                black_king,
            );
            self.removeFeature(acc, cap_feature);
        }

        // Handle promotion
        if (move.promotion) |promo_kind| {
            // Add promoted piece at new square
            const promo_feature = features.getFeatureWithKings(
                promo_kind,
                moving_piece.color,
                move.to,
                white_king,
                black_king,
            );
            self.addFeature(acc, promo_feature);
        } else {
            // Add piece at new square
            const new_feature = features.getFeatureWithKings(
                moving_piece.kind,
                moving_piece.color,
                move.to,
                white_king,
                black_king,
            );
            self.addFeature(acc, new_feature);
        }

        return true;
    }

    /// Build a child accumulator directly from its parent in one SIMD pass.
    /// Returns false for moves which require a full refresh.
    pub fn updateMoveFromParent(
        self: *const AccumulatorUpdater,
        parent: *const Accumulator,
        child: *Accumulator,
        state: *const GameState,
        move: Move,
        captured_piece: ?PieceKind,
    ) bool {
        const moving_piece = state.board.squares[move.to].piece orelse return false;
        if (moving_piece.kind == .King or move.is_castle or move.is_en_passant or move.promotion != null) {
            return false;
        }

        const white_king = state.king_squares[0];
        const black_king = state.king_squares[1];
        const old_feature = features.getFeatureWithKings(
            moving_piece.kind,
            moving_piece.color,
            move.from,
            white_king,
            black_king,
        );
        const new_feature = features.getFeatureWithKings(
            moving_piece.kind,
            moving_piece.color,
            move.to,
            white_king,
            black_king,
        );
        const captured_feature: ?features.Feature = if (captured_piece) |kind|
            features.getFeatureWithKings(
                kind,
                1 - moving_piece.color,
                move.to,
                white_king,
                black_king,
            )
        else
            null;

        var i: usize = 0;
        while (i < HIDDEN_SIZE) : (i += SIMD_LANES) {
            const parent_white: VecI16 = @bitCast(parent.white[i..][0..SIMD_LANES].*);
            const parent_black: VecI16 = @bitCast(parent.black[i..][0..SIMD_LANES].*);
            const old_white: VecI16 = @bitCast(self.input_weights[old_feature.index_white][i..][0..SIMD_LANES].*);
            const old_black: VecI16 = @bitCast(self.input_weights[old_feature.index_black][i..][0..SIMD_LANES].*);
            const new_white: VecI16 = @bitCast(self.input_weights[new_feature.index_white][i..][0..SIMD_LANES].*);
            const new_black: VecI16 = @bitCast(self.input_weights[new_feature.index_black][i..][0..SIMD_LANES].*);

            var child_white = parent_white -% old_white;
            var child_black = parent_black -% old_black;
            if (captured_feature) |captured| {
                const captured_white: VecI16 = @bitCast(self.input_weights[captured.index_white][i..][0..SIMD_LANES].*);
                const captured_black: VecI16 = @bitCast(self.input_weights[captured.index_black][i..][0..SIMD_LANES].*);
                child_white -%= captured_white;
                child_black -%= captured_black;
            }
            child.white[i..][0..SIMD_LANES].* = @bitCast(child_white +% new_white);
            child.black[i..][0..SIMD_LANES].* = @bitCast(child_black +% new_black);
        }
        return true;
    }
};
