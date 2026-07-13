const std = @import("std");
const Network = @import("network.zig").Network;
const INPUT_SIZE = @import("network.zig").INPUT_SIZE;
const HIDDEN_SIZE = @import("network.zig").HIDDEN_SIZE;
const HIDDEN_DSIZE = @import("network.zig").HIDDEN_DSIZE;
const OUTPUT_SIZE = @import("network.zig").OUTPUT_SIZE;

pub fn loadFromFile(network: *Network, path: []const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var buffered = std.io.bufferedReader(file.reader());
    var reader = buffered.reader();

    std.debug.print("Loading NNUE weights from: {s}\n", .{path});
    std.debug.print("Network architecture: ({d} -> {d})x2 -> {d}\n", .{ INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE });

    std.debug.print("Loading input weights ({d} x {d})...\n", .{ INPUT_SIZE, HIDDEN_SIZE });
    for (network.input_weights) |*weight_row| {
        for (weight_row) |*weight| {
            weight.* = try readI16LE(&reader);
        }
    }

    std.debug.print("Loading input biases ({d})...\n", .{HIDDEN_SIZE});
    for (network.input_biases) |*bias| {
        bias.* = try readI16LE(&reader);
    }

    std.debug.print("Loading hidden weights ({d} x {d})...\n", .{ OUTPUT_SIZE, HIDDEN_DSIZE });
    for (network.hidden_weights) |*weight_row| {
        for (weight_row) |*weight| {
            weight.* = try readI16LE(&reader);
        }
    }

    std.debug.print("Loading hidden biases ({d})...\n", .{OUTPUT_SIZE});
    for (network.hidden_biases) |*bias| {
        bias.* = try readI32LE(&reader);
    }

    std.debug.print("NNUE weights loaded successfully!\n", .{});

    network.loaded = true;

    network.updater = @import("accumulator.zig").AccumulatorUpdater.init(
        network.input_weights,
        network.input_biases,
    );
}

fn readI16LE(reader: anytype) !i16 {
    const bytes = try reader.readBytesNoEof(2);
    const unsigned = @as(u16, bytes[0]) | (@as(u16, bytes[1]) << 8);
    return @bitCast(unsigned);
}

fn readI32LE(reader: anytype) !i32 {
    const bytes = try reader.readBytesNoEof(4);
    const unsigned = @as(u32, bytes[0]) |
        (@as(u32, bytes[1]) << 8) |
        (@as(u32, bytes[2]) << 16) |
        (@as(u32, bytes[3]) << 24);
    return @bitCast(unsigned);
}

pub fn verifyFile(path: []const u8) !bool {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        std.debug.print("Could not open NNUE file: {}\n", .{err});
        return false;
    };
    defer file.close();

    const stat = try file.stat();

    const expected_size = INPUT_SIZE * HIDDEN_SIZE * 2 +
        HIDDEN_SIZE * 2 +
        OUTPUT_SIZE * HIDDEN_DSIZE * 2 +
        OUTPUT_SIZE * 4;

    std.debug.print("File size: {d} bytes, Expected: {d} bytes\n", .{ stat.size, expected_size });

    if (stat.size != expected_size) {
        std.debug.print("Warning: File size mismatch! Expected {d}, got {d}\n", .{ expected_size, stat.size });
        return false;
    }

    return true;
}

pub fn loadNetwork(network: *Network, path: []const u8) bool {
    loadFromFile(network, path) catch |err| {
        std.debug.print("Failed to load NNUE network: {}\n", .{err});
        std.debug.print("Falling back to classical evaluation.\n", .{});
        return false;
    };
    return true;
}
