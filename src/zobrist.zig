const std = @import("std");
const PieceKind = @import("piece.zig").PieceKind;

/// Zobrist hashing for position identification
/// Each unique position gets a unique 64-bit hash (with high probability)

// Zobrist random numbers for hashing
// [piece_type][color][square]
var PIECE_KEYS: [6][2][64]u64 = undefined;
var CASTLE_KEYS: [16]u64 = undefined; // All combinations of castling rights
var EP_KEYS: [8]u64 = undefined; // En passant file (0-7)
var SIDE_KEY: u64 = undefined; // Black to move

var initialized = false;

/// Initialize Zobrist keys with pseudo-random numbers
pub fn init() void {
    if (initialized) return;
    
    // Use a fixed seed for reproducibility (important for debugging)
    var prng = std.Random.DefaultPrng.init(0x1234567890ABCDEF);
    var rand = prng.random();
    
    // Generate piece keys
    for (&PIECE_KEYS) |*piece_type| {
        for (piece_type) |*color| {
            for (color) |*square_key| {
                square_key.* = rand.int(u64);
            }
        }
    }
    
    // Generate castling keys
    for (&CASTLE_KEYS) |*key| {
        key.* = rand.int(u64);
    }
    
    // Generate en passant keys
    for (&EP_KEYS) |*key| {
        key.* = rand.int(u64);
    }
    
    // Generate side key
    SIDE_KEY = rand.int(u64);
    
    initialized = true;
}

/// Get the Zobrist key for a piece on a square
pub inline fn getPieceKey(piece_kind: PieceKind, color: i8, square: u8) u64 {
    const piece_idx = @intFromEnum(piece_kind);
    const color_idx: usize = if (color == 1) 0 else 1;
    return PIECE_KEYS[piece_idx][color_idx][square];
}

/// Get the Zobrist key for castling rights
pub inline fn getCastlingKey(rights: u8) u64 {
    return CASTLE_KEYS[rights & 0x0F];
}

/// Get the Zobrist key for en passant file
pub inline fn getEnPassantKey(file: u8) u64 {
    return EP_KEYS[file];
}

/// Get the Zobrist key for side to move (black)
pub inline fn getSideKey() u64 {
    return SIDE_KEY;
}

/// Encode castling rights as a 4-bit value
pub fn encodeCastlingRights(white_kingside: bool, white_queenside: bool, black_kingside: bool, black_queenside: bool) u8 {
    var rights: u8 = 0;
    if (white_kingside) rights |= 0x1;
    if (white_queenside) rights |= 0x2;
    if (black_kingside) rights |= 0x4;
    if (black_queenside) rights |= 0x8;
    return rights;
}
