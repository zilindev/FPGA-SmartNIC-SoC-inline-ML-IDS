// register_file.v — 16x64-bit FF-based, 3 read ports, 1 write port
`timescale 1ns / 1ps
`include "gpu_params.vh"

module register_file (
    input                       clk,
    input                       rst,

    input  [31:0]               thread_id,
    input  [31:0]               block_id,
    input  [31:0]               block_dim,

    // Write port
    input                       wr_en,
    input  [`REG_ADDR_W-1:0]    wr_addr,
    input  [`DATA_WIDTH-1:0]    wr_data,

    // Read ports
    input  [`REG_ADDR_W-1:0]    rd_addr1,
    output [`DATA_WIDTH-1:0]    rd_data1,
    input  [`REG_ADDR_W-1:0]    rd_addr2,
    output [`DATA_WIDTH-1:0]    rd_data2,
    input  [`REG_ADDR_W-1:0]    rd_addr3,
    output [`DATA_WIDTH-1:0]    rd_data3
);

    reg [`DATA_WIDTH-1:0] regs [0:`REG_COUNT-1];

    integer k;

    // Synchronous write (R0, R13-R15 are read-only)
    always @(posedge clk) begin
        if (rst) begin
            for (k = 0; k < `REG_COUNT; k = k + 1)
                regs[k] <= {`DATA_WIDTH{1'b0}};
        end else if (wr_en && wr_addr != `REG_ZERO
                            && wr_addr != `REG_THREADID
                            && wr_addr != `REG_BLOCKID
                            && wr_addr != `REG_BLOCKDIM) begin
            regs[wr_addr] <= wr_data;
        end
    end

    // Async read with special register overrides
    assign rd_data1 = (rd_addr1 == `REG_ZERO)     ? {`DATA_WIDTH{1'b0}} :
                      (rd_addr1 == `REG_THREADID)  ? {{32{1'b0}}, thread_id} :
                      (rd_addr1 == `REG_BLOCKID)   ? {{32{1'b0}}, block_id} :
                      (rd_addr1 == `REG_BLOCKDIM)  ? {{32{1'b0}}, block_dim} :
                      regs[rd_addr1];

    assign rd_data2 = (rd_addr2 == `REG_ZERO)     ? {`DATA_WIDTH{1'b0}} :
                      (rd_addr2 == `REG_THREADID)  ? {{32{1'b0}}, thread_id} :
                      (rd_addr2 == `REG_BLOCKID)   ? {{32{1'b0}}, block_id} :
                      (rd_addr2 == `REG_BLOCKDIM)  ? {{32{1'b0}}, block_dim} :
                      regs[rd_addr2];

    assign rd_data3 = (rd_addr3 == `REG_ZERO)     ? {`DATA_WIDTH{1'b0}} :
                      (rd_addr3 == `REG_THREADID)  ? {{32{1'b0}}, thread_id} :
                      (rd_addr3 == `REG_BLOCKID)   ? {{32{1'b0}}, block_id} :
                      (rd_addr3 == `REG_BLOCKDIM)  ? {{32{1'b0}}, block_dim} :
                      regs[rd_addr3];

endmodule
