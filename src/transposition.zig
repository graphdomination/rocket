const std = @import("std");
const Move = @import("movegen.zig").Move;
const UndoInfo = @import("movegen.zig").UndoInfo;
const GameState = @import("gamestate.zig").GameState;
const Piece = @import("piece.zig").Piece;
const zobrist = @import("zobrist.zig");

/// Entry type in the transposition table
pub const EntryType = enum(u8) {
    Exact = 0, // PV-node (exact score)
    Alpha = 1, // All-node (upper bound, failed low)
    Beta = 2, // Cut-node (lower bound, failed high)
};

/// Transposition table entry
pub const TTEntry = struct {
    hash: u64, // Full hash for verification
    best_move: Move, // Best move found in this position
    score: i32, // Evaluation score
    depth: i8, // Search depth
    entry_type: EntryType, // Type of node
    age: u8, // Search age (for replacement)

    pub fn isEmpty(self: *const TTEntry) bool {
        return self.hash == 0;
    }
};

/// A lock-free table slot. The entry payload and its 16-bit hash signature fit
/// in one atomic word, so concurrent writers can never publish a torn entry.
const AtomicTTEntry = struct {
    data: std.atomic.Value(u64),

    fn init() AtomicTTEntry {
        return .{
            .data = std.atomic.Value(u64).init(0),
        };
    }
};

const Snapshot = struct {
    signature: u16,
    entry: TTEntry,
};

/// Transposition table
pub const TranspositionTable = struct {
    entries: []AtomicTTEntry,
    allocator: std.mem.Allocator,
    size: usize,
    age: u8, // Current search age

    /// Create a new transposition table with given size in MB
    pub fn init(allocator: std.mem.Allocator, size_mb: usize) !TranspositionTable {
        const bytes = size_mb * 1024 * 1024;
        const entry_size = @sizeOf(AtomicTTEntry);
        const num_entries = bytes / entry_size;

        const entries = try allocator.alloc(AtomicTTEntry, num_entries);
        for (entries) |*entry| entry.* = AtomicTTEntry.init();

        return TranspositionTable{
            .entries = entries,
            .allocator = allocator,
            .size = num_entries,
            .age = 0,
        };
    }

    pub fn deinit(self: *TranspositionTable) void {
        self.allocator.free(self.entries);
    }

    /// Clear all entries
    pub fn clear(self: *TranspositionTable) void {
        for (self.entries) |*entry| {
            entry.data.store(0, .release);
        }
        self.age = 0;
    }

    /// Increment search age (call at start of new search)
    pub fn incrementAge(self: *TranspositionTable) void {
        self.age +%= 1; // Wrapping add
    }

    /// Get the index for a hash
    inline fn getIndex(self: *const TranspositionTable, hash: u64) usize {
        return @intCast((@as(u128, hash) * @as(u128, self.size)) >> 64);
    }

    /// Probe the transposition table
    pub fn probe(self: *const TranspositionTable, hash: u64) ?TTEntry {
        const index = self.getIndex(hash);
        const snapshot = readSnapshot(&self.entries[index]) orelse return null;
        if (snapshot.signature == hashSignature(hash)) {
            var entry = snapshot.entry;
            entry.hash = hash;
            return entry;
        }
        return null;
    }

    /// Store an entry in the transposition table
    pub fn store(
        self: *TranspositionTable,
        hash: u64,
        best_move: Move,
        score: i32,
        depth: i8,
        entry_type: EntryType,
    ) void {
        const index = self.getIndex(hash);
        const slot = &self.entries[index];
        const current = readSnapshot(slot);
        const signature = hashSignature(hash);
        const should_replace = current == null or
            current.?.signature == signature or
            depth > current.?.entry.depth or
            (depth == current.?.entry.depth and self.age != current.?.entry.age) or
            (self.age -% current.?.entry.age) >= 2;

        if (should_replace) {
            const payload = packEntry(TTEntry{
                .hash = hash,
                .best_move = best_move,
                .score = score,
                .depth = depth,
                .entry_type = entry_type,
                .age = self.age,
            });
            const encoded = payload | (@as(u64, signature) << 48);
            slot.data.store(encoded, .release);
        }
    }

    /// Get fill percentage (for debugging/info)
    pub fn getFillPercentage(self: *const TranspositionTable) f64 {
        var filled: usize = 0;
        const sample_size = @min(1000, self.size);

        for (self.entries[0..sample_size]) |*entry| {
            if (entry.data.load(.monotonic) != 0) {
                filled += 1;
            }
        }

        return @as(f64, @floatFromInt(filled)) / @as(f64, @floatFromInt(sample_size)) * 100.0;
    }
};

inline fn hashSignature(hash: u64) u16 {
    // getIndex uses the high part of hash * table_size. For power-of-two table
    // sizes (including every normal MB setting), the old top-16 signature was
    // fully determined by the bucket index and verified nothing at all. Use
    // independent low bits so same-bucket collisions cannot masquerade as hits.
    return @truncate(hash);
}

fn packEntry(entry: TTEntry) u64 {
    const promotion: u64 = if (entry.best_move.promotion) |kind|
        @as(u64, @intFromEnum(kind)) + 1
    else
        0;
    const bounded_score: i16 = @intCast(std.math.clamp(entry.score, -32000, 32000));
    const score_bits: u16 = @bitCast(bounded_score);
    const depth_bits: u8 = @bitCast(entry.depth);

    return @as(u64, entry.best_move.from) |
        (@as(u64, entry.best_move.to) << 6) |
        (promotion << 12) |
        (@as(u64, score_bits) << 15) |
        ((@as(u64, depth_bits) & 0x7f) << 31) |
        (@as(u64, @intFromEnum(entry.entry_type)) << 38) |
        (@as(u64, entry.age) << 40);
}

fn unpackEntry(hash: u64, data: u64) TTEntry {
    const promotion_code: u3 = @truncate(data >> 12);
    const score_bits: u16 = @truncate(data >> 15);
    const depth_bits: u7 = @truncate(data >> 31);
    return .{
        .hash = hash,
        .best_move = .{
            .from = @intCast(data & 0x3f),
            .to = @intCast((data >> 6) & 0x3f),
            .promotion = if (promotion_code == 0)
                null
            else
                @enumFromInt(promotion_code - 1),
        },
        .score = @as(i16, @bitCast(score_bits)),
        .depth = @intCast(depth_bits),
        .entry_type = @enumFromInt(@as(u2, @truncate(data >> 38))),
        .age = @truncate(data >> 40),
    };
}

fn readSnapshot(slot: *const AtomicTTEntry) ?Snapshot {
    const data = slot.data.load(.acquire);
    if (data == 0) return null;
    return .{
        .signature = @truncate(data >> 48),
        .entry = unpackEntry(0, data),
    };
}

test "packed transposition entry round trips move and score" {
    const original = TTEntry{
        .hash = 0x123456789abcdef0,
        .best_move = .{ .from = 60, .to = 62, .promotion = .Queen },
        .score = -1234,
        .depth = 27,
        .entry_type = .Beta,
        .age = 201,
    };
    const decoded = unpackEntry(original.hash, packEntry(original));
    try std.testing.expectEqual(original.hash, decoded.hash);
    try std.testing.expect(decoded.best_move.equals(original.best_move));
    try std.testing.expectEqual(original.score, decoded.score);
    try std.testing.expectEqual(original.depth, decoded.depth);
    try std.testing.expectEqual(original.entry_type, decoded.entry_type);
    try std.testing.expectEqual(original.age, decoded.age);
}

test "same-bucket collision fails signature verification" {
    var table = try TranspositionTable.init(std.testing.allocator, 1);
    defer table.deinit();

    const stored_hash: u64 = 0x1234_5678_9abc_0000;
    const colliding_hash = stored_hash ^ 1;
    try std.testing.expectEqual(table.getIndex(stored_hash), table.getIndex(colliding_hash));

    const best_move = Move{ .from = 12, .to = 28 };
    table.store(stored_hash, best_move, 137, 12, .Exact);

    const hit = table.probe(stored_hash) orelse return error.ExpectedStoredHit;
    try std.testing.expect(hit.best_move.equals(best_move));
    try std.testing.expect(table.probe(colliding_hash) == null);
}

/// Compute Zobrist hash for a position
pub fn computeHash(state: *const GameState) u64 {
    var hash: u64 = 0;

    // Hash pieces
    for (state.board.squares, 0..) |square, idx| {
        if (square.piece) |piece| {
            hash ^= zobrist.getPieceKey(piece.kind, piece.color, @intCast(idx));
        }
    }

    // Hash castling rights
    const rights = zobrist.encodeCastlingRights(
        state.castling_rights.white_kingside,
        state.castling_rights.white_queenside,
        state.castling_rights.black_kingside,
        state.castling_rights.black_queenside,
    );
    hash ^= zobrist.getCastlingKey(rights);

    // Hash en passant
    if (state.en_passant_square) |ep| {
        const file = ep % 8;
        hash ^= zobrist.getEnPassantKey(file);
    }

    // Hash side to move
    if (state.side_to_move == 0) { // Black to move
        hash ^= zobrist.getSideKey();
    }

    return hash;
}

/// Incrementally update a position key after makeMove has been applied.
pub fn hashAfterMove(
    old_hash: u64,
    state: *const GameState,
    move: Move,
    undo: *const UndoInfo,
    moving_piece: Piece,
) u64 {
    var hash = old_hash;

    hash ^= zobrist.getPieceKey(moving_piece.kind, moving_piece.color, move.from);
    if (undo.captured_piece) |captured| {
        const captured_square = if (move.is_en_passant)
            (if (moving_piece.color == 1) move.to + 8 else move.to - 8)
        else
            move.to;
        hash ^= zobrist.getPieceKey(captured.kind, captured.color, captured_square);
    }
    const placed_kind = move.promotion orelse moving_piece.kind;
    hash ^= zobrist.getPieceKey(placed_kind, moving_piece.color, move.to);

    if (move.is_castle) {
        const rook_from: u8 = switch (move.to) {
            62 => 63,
            58 => 56,
            6 => 7,
            2 => 0,
            else => unreachable,
        };
        const rook_to: u8 = switch (move.to) {
            62 => 61,
            58 => 59,
            6 => 5,
            2 => 3,
            else => unreachable,
        };
        hash ^= zobrist.getPieceKey(.Rook, moving_piece.color, rook_from);
        hash ^= zobrist.getPieceKey(.Rook, moving_piece.color, rook_to);
    }

    const old_rights = zobrist.encodeCastlingRights(
        undo.castling_rights.white_kingside,
        undo.castling_rights.white_queenside,
        undo.castling_rights.black_kingside,
        undo.castling_rights.black_queenside,
    );
    const new_rights = zobrist.encodeCastlingRights(
        state.castling_rights.white_kingside,
        state.castling_rights.white_queenside,
        state.castling_rights.black_kingside,
        state.castling_rights.black_queenside,
    );
    hash ^= zobrist.getCastlingKey(old_rights);
    hash ^= zobrist.getCastlingKey(new_rights);

    if (undo.en_passant_square) |ep| hash ^= zobrist.getEnPassantKey(ep % 8);
    if (state.en_passant_square) |ep| hash ^= zobrist.getEnPassantKey(ep % 8);
    hash ^= zobrist.getSideKey();
    return hash;
}

pub fn hashAfterNull(old_hash: u64, old_ep_square: ?u8) u64 {
    var hash = old_hash ^ zobrist.getSideKey();
    if (old_ep_square) |ep| hash ^= zobrist.getEnPassantKey(ep % 8);
    return hash;
}

/// Update hash incrementally after a move (more efficient than recomputing)
pub fn updateHashAfterMove(
    hash: u64,
    state: *const GameState,
    move: Move,
    old_castling_rights: u8,
    old_ep_square: ?u8,
) u64 {
    // This is a placeholder for incremental update
    // For simplicity, we'll recompute the hash
    // In a competition engine, you'd want true incremental updates
    _ = hash;
    _ = move;
    _ = old_castling_rights;
    _ = old_ep_square;

    return computeHash(state);
}
