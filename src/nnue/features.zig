const std = @import("std");
const GameState = @import("../gamestate.zig").GameState;
const Piece = @import("../piece.zig").Piece;
const PieceKind = @import("../piece.zig").PieceKind;

/// NNUE Feature Set: HalfKP (Half King-Piece)
/// Features are defined relative to the king position
///
/// Architecture constants matching Koivisto format:
/// - 6 piece types (P, N, B, R, Q, K)
/// - 64 squares
/// - 2 colors
/// - 16 king buckets with horizontal mirroring
/// - Input size: 6 * 64 * 2 * 16 = 12,288
pub const N_PIECE_TYPES: usize = 6; // P, N, B, R, Q, K
pub const N_SQUARES: usize = 64;
pub const N_COLORS: usize = 2;
pub const N_KING_BUCKETS: usize = 16;
pub const INPUT_SIZE: usize = N_PIECE_TYPES * N_SQUARES * N_COLORS * N_KING_BUCKETS; // 12,288

/// King square bucketing table (16 buckets with horizontal mirroring)
/// Based on Koivisto's king square indices
pub const KING_SQUARE_INDICES = [64]u8{
    0,  1,  2,  3,  3,  2,  1,  0,
    4,  5,  6,  7,  7,  6,  5,  4,
    8,  9,  10, 11, 11, 10, 9,  8,
    8,  9,  10, 11, 11, 10, 9,  8,
    12, 12, 13, 13, 13, 13, 12, 12,
    12, 12, 13, 13, 13, 13, 12, 12,
    14, 14, 15, 15, 15, 15, 14, 14,
    14, 14, 15, 15, 15, 15, 14, 14,
};

/// Feature representation for a piece on the board
pub const Feature = struct {
    index_white: usize, // Feature index from white's perspective
    index_black: usize, // Feature index from black's perspective
};

/// Get king square index (bucket 0-15)
pub fn getKingSquareIndex(king_square: u8, king_color: i8) u8 {
    // Rocket numbers A8 as square 0, while the Koivisto network was trained
    // with A1 as square 0. Convert to the network's coordinate system first.
    var sq = king_square ^ 56;
    // Then orient the board for Black's perspective.
    if (king_color == 0) {
        sq ^= 56;
    }
    return KING_SQUARE_INDICES[sq];
}

/// Get feature index for a piece from a given perspective
///
/// Arguments:
/// - piece_kind: Type of piece (P, N, B, R, Q, K)
/// - piece_color: Color of the piece (1 = white, 0 = black)
/// - square: Square where piece is located (0-63)
/// - view: Perspective color (1 = white's view, 0 = black's view)
/// - king_square: Square where the king of 'view' color is located
///
/// Index calculation:
/// index = square + piece_type*64 + color_idx*64*6 + king_bucket*64*6*2
///
/// Where:
/// - square is flipped based on view
/// - horizontal flip if king is on right half
/// - color_idx is 1 for friendly pieces, 0 for enemy pieces
/// - king_bucket is based on king position (0-15)
pub fn getFeatureIndex(
    piece_kind: PieceKind,
    piece_color: i8,
    square: u8,
    view: i8,
    king_square: u8,
) usize {
    const ks_index = getKingSquareIndex(king_square, view);
    // Convert Rocket's A8-origin squares to Koivisto's A1-origin squares.
    var sq = square ^ 56;

    // Flip square based on view (rank flip).
    if (view == 0) { // Black's perspective
        sq ^= 56;
    }

    // Horizontal flip if king is on right half of board
    if ((king_square & 0x4) != 0) {
        sq ^= 7;
    }

    // Piece type index (0-5 for P, N, B, R, Q, K)
    const pt = @intFromEnum(piece_kind);

    // Color from perspective:
    // 1 if colors are same (friendly), 0 if different (enemy)
    const color_idx: usize = if (piece_color == view) 1 else 0;

    // Calculate index: square + piece_type*64 + color_idx*64*6 + king_bucket*64*6*2
    const index = @as(usize, sq) +
        @as(usize, pt) * 64 +
        color_idx * 384 +
        @as(usize, ks_index) * 768;

    return index;
}

test "Koivisto feature coordinates use A1 origin" {
    // White pawn e2 with kings on e1/e8 in Rocket's A8-origin coordinates.
    const feature = getFeatureWithKings(.Pawn, 1, 52, 60, 4);
    // The e-file king also activates Koivisto's horizontal mirroring.
    try std.testing.expectEqual(@as(usize, 2699), feature.index_white);
    try std.testing.expectEqual(@as(usize, 2355), feature.index_black);
}

/// Get both feature indices (white and black perspective) for a piece
pub fn getFeature(
    state: *const GameState,
    piece_kind: PieceKind,
    piece_color: i8,
    square: u8,
) Feature {
    // Find kings
    const white_king = findKing(state, 1) orelse return Feature{ .index_white = 0, .index_black = 0 };
    const black_king = findKing(state, 0) orelse return Feature{ .index_white = 0, .index_black = 0 };

    return Feature{
        .index_white = getFeatureIndex(piece_kind, piece_color, square, 1, white_king),
        .index_black = getFeatureIndex(piece_kind, piece_color, square, 0, black_king),
    };
}

/// Build feature indices when the caller already knows both king squares.
/// Hot NNUE update paths use this to avoid repeatedly scanning all 64 squares.
pub fn getFeatureWithKings(
    piece_kind: PieceKind,
    piece_color: i8,
    square: u8,
    white_king: u8,
    black_king: u8,
) Feature {
    return Feature{
        .index_white = getFeatureIndex(piece_kind, piece_color, square, 1, white_king),
        .index_black = getFeatureIndex(piece_kind, piece_color, square, 0, black_king),
    };
}

/// Find the king of a given color on the board
pub fn findKing(state: *const GameState, color: i8) ?u8 {
    for (state.board.squares, 0..) |square, idx| {
        if (square.piece) |piece| {
            if (piece.kind == .King and piece.color == color) {
                return @intCast(idx);
            }
        }
    }
    return null;
}

/// Get all active features for a position
/// Returns a list of features (up to all 32 pieces, including both kings)
pub fn getActiveFeatures(state: *const GameState, allocator: std.mem.Allocator) ![]Feature {
    var features = std.ArrayList(Feature).init(allocator);

    for (state.board.squares, 0..) |square, idx| {
        if (square.piece) |piece| {
            const feature = getFeature(state, piece.kind, piece.color, @intCast(idx));
            try features.append(feature);
        }
    }

    return features.toOwnedSlice();
}
