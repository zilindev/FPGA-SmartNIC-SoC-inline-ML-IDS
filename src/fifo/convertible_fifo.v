`timescale 1ns / 1ps

// convertible_fifo — Packet-buffering FIFO with processor access

module convertible_fifo (
    input         clk,
    input         reset,

    // NF2.1 input
    input  [63:0] in_data,
    input  [7:0]  in_ctrl,
    input         in_wr,
    output        in_rdy,

    // NF2.1 output
    output [63:0] out_data,
    output [7:0]  out_ctrl,
    output        out_wr,
    input         out_rdy,

    // Processor access port (CPU or GPU via mux)
    input  [7:0]  proc_addr,
    input  [71:0] proc_din,
    input         proc_we,
    output [71:0] proc_dout,

    // Control
    input  [1:0]  mode,         // 0=FIFO, 1=CPU, 2=GPU
    input         drain_start,

    // Status
    output        pkt_ready,
    output        fifo_full,
    output        fifo_empty,
    output [7:0]  pkt_word_cnt,

    // Head/tail register access
    input  [7:0]  head_wr_val,
    input         head_wr_en,
    input  [7:0]  tail_wr_val,
    input         tail_wr_en,
    output [7:0]  head_rd,
    output [7:0]  tail_rd
);

    // --- State machine ---
    localparam S_IDLE          = 3'd0;
    localparam S_RECEIVING     = 3'd1;
    localparam S_PKT_BUFFERED  = 3'd2;
    localparam S_DRAIN_WAIT1   = 3'd3;
    localparam S_DRAIN_WAIT2   = 3'd4;
    localparam S_DRAINING      = 3'd5;

    reg [2:0] state;

    // --- Pointers and counters ---
    reg [7:0] wr_ptr;
    reg [7:0] rd_ptr;
    reg [7:0] pkt_len;
    reg [7:0] words_remaining;
    reg       pkt_stored;
    reg [7:0] drain_cnt;

    assign head_rd      = rd_ptr;
    assign tail_rd      = wr_ptr;
    assign pkt_ready    = pkt_stored;
    assign fifo_full    = pkt_stored;
    assign fifo_empty   = (state == S_IDLE) && !pkt_stored;
    assign pkt_word_cnt = pkt_len;

    // Only accept input in FIFO mode
    assign in_rdy = (mode == 2'd0)
                 && (state == S_IDLE || state == S_RECEIVING)
                 && !pkt_stored;

    // --- BRAM port signals ---
    reg  [7:0]  bram_addr_a;
    reg  [71:0] bram_din_a;
    reg         bram_we_a;
    wire [71:0] bram_dout_a;

    reg  [7:0]  bram_addr_b;
    reg  [71:0] bram_din_b;
    reg         bram_we_b;
    wire [71:0] bram_dout_b;

    fifo_bram u_bram (
        .clk    (clk),
        .addr_a (bram_addr_a),
        .din_a  (bram_din_a),
        .we_a   (bram_we_a),
        .dout_a (bram_dout_a),
        .addr_b (bram_addr_b),
        .din_b  (bram_din_b),
        .we_b   (bram_we_b),
        .dout_b (bram_dout_b)
    );

    // --- Port A: FIFO write logic ---
    wire fifo_wr_active = in_wr && in_rdy;

    always @(*) begin
        if (fifo_wr_active) begin
            bram_addr_a = wr_ptr;
            bram_din_a  = {in_ctrl, in_data};
            bram_we_a   = 1'b1;
        end else begin
            bram_addr_a = 8'd0;
            bram_din_a  = 72'd0;
            bram_we_a   = 1'b0;
        end
    end

    // --- Port B mux: processor or FIFO read ---
    always @(*) begin
        case (mode)
            2'd1, 2'd2: begin
                bram_addr_b = proc_addr;
                bram_din_b  = proc_din;
                bram_we_b   = proc_we;
            end
            default: begin
                bram_addr_b = rd_ptr;
                bram_din_b  = 72'd0;
                bram_we_b   = 1'b0;
            end
        endcase
    end

    assign proc_dout = bram_dout_b;

    // --- Drain output ---
    reg        drain_data_valid;
    reg [71:0] drain_word_reg;

    assign out_data = drain_word_reg[63:0];
    assign out_ctrl = drain_word_reg[71:64];
    assign out_wr   = drain_data_valid && (state == S_DRAINING);

    // --- State machine ---
    always @(posedge clk) begin
        if (reset) begin
            state           <= S_IDLE;
            wr_ptr          <= 8'd0;
            rd_ptr          <= 8'd0;
            pkt_len         <= 8'd0;
            words_remaining <= 8'd0;
            pkt_stored      <= 1'b0;
            drain_data_valid<= 1'b0;
            drain_word_reg  <= 72'd0;
            drain_cnt       <= 8'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    drain_data_valid <= 1'b0;
                    if (fifo_wr_active && in_ctrl != 8'd0) begin
                        wr_ptr  <= wr_ptr + 8'd1;
                        pkt_len <= in_data[39:32];
                        if (in_data[39:32] <= 8'd1) begin
                            words_remaining <= 8'd0;
                            pkt_stored      <= 1'b1;
                            state           <= S_PKT_BUFFERED;
                        end else begin
                            words_remaining <= in_data[39:32] - 8'd1;
                            state           <= S_RECEIVING;
                        end
                    end
                end

                S_RECEIVING: begin
                    if (fifo_wr_active) begin
                        wr_ptr          <= wr_ptr + 8'd1;
                        words_remaining <= words_remaining - 8'd1;
                        if (words_remaining == 8'd1) begin
                            pkt_stored <= 1'b1;
                            state      <= S_PKT_BUFFERED;
                        end
                    end
                end

                S_PKT_BUFFERED: begin
                    if (head_wr_en) rd_ptr <= head_wr_val;
                    if (tail_wr_en) wr_ptr <= tail_wr_val;

                    if (drain_start) begin
                        rd_ptr    <= 8'd0;
                        drain_cnt <= 8'd0;
                        state     <= S_DRAIN_WAIT1;
                    end
                end

                S_DRAIN_WAIT1: begin
                    rd_ptr <= rd_ptr + 8'd1;
                    state  <= S_DRAIN_WAIT2;
                end

                S_DRAIN_WAIT2: begin
                    drain_word_reg   <= bram_dout_b;
                    drain_data_valid <= 1'b1;
                    rd_ptr           <= rd_ptr + 8'd1;
                    state            <= S_DRAINING;
                end

                S_DRAINING: begin
                    if (out_rdy && drain_data_valid) begin
                        drain_cnt <= drain_cnt + 8'd1;
                        if (drain_cnt + 8'd1 == pkt_len) begin
                            drain_data_valid <= 1'b0;
                            pkt_stored       <= 1'b0;
                            wr_ptr           <= 8'd0;
                            rd_ptr           <= 8'd0;
                            pkt_len          <= 8'd0;
                            drain_cnt        <= 8'd0;
                            state            <= S_IDLE;
                        end else begin
                            drain_word_reg <= bram_dout_b;
                            rd_ptr         <= rd_ptr + 8'd1;
                        end
                    end
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
