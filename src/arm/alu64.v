`timescale 1ns / 1ps

// alu64.v -- ARM ALU (combinational), structural addsub64 + logic64

module alu64 (
    input  [63:0] A,
    input  [63:0] B,
    input  [3:0]  alu_ctrl, // ARM opcode [24:21]
    output [63:0] Z,
    output        N_flag,
    output        Z_flag,
    output        C_flag,
    output        V_flag
);

    // Arithmetic unit
    wire is_sub = (alu_ctrl == 4'b0010) || // SUB
                  (alu_ctrl == 4'b1010);   // CMP

    wire [63:0] arith_result;
    wire        arith_cout;

    addsub64 u_addsub (
        .A(A),
        .B(B),
        .sub_mode(is_sub),
        .S(arith_result),
        .overflow(arith_cout)
    );

    // Signed overflow
    wire b_eff_msb = B[63] ^ is_sub;
    wire arith_V   = (~(A[63] ^ b_eff_msb)) & (arith_result[63] ^ A[63]);

    // Logic unit
    wire [63:0] and_out, or_out, xnor_out;

    logic64 u_logic (
        .A(A),
        .B(B),
        .AND_out(and_out),
        .OR_out(or_out),
        .XNOR_out(xnor_out)
    );

    wire [63:0] eor_out = A ^ B;
    wire [63:0] bic_out = A & (~B);
    wire [63:0] mvn_out = ~B;

    // Output mux
    reg [63:0] mux_Z;
    reg        mux_C;

    always @(*) begin
        mux_Z = arith_result;
        mux_C = arith_cout;

        case (alu_ctrl)
            4'b0100: begin mux_Z = arith_result; mux_C = arith_cout; end // ADD
            4'b0010: begin mux_Z = arith_result; mux_C = arith_cout; end // SUB
            4'b1010: begin mux_Z = arith_result; mux_C = arith_cout; end // CMP
            4'b0000: begin mux_Z = and_out;      mux_C = 1'b0;      end // AND
            4'b0001: begin mux_Z = eor_out;      mux_C = 1'b0;      end // EOR
            4'b1000: begin mux_Z = and_out;      mux_C = 1'b0;      end // TST
            4'b1001: begin mux_Z = eor_out;      mux_C = 1'b0;      end // TEQ
            4'b1100: begin mux_Z = or_out;       mux_C = 1'b0;      end // ORR
            4'b1110: begin mux_Z = bic_out;      mux_C = 1'b0;      end // BIC
            4'b1111: begin mux_Z = mvn_out;      mux_C = 1'b0;      end // MVN
            4'b1101: begin mux_Z = B;            mux_C = 1'b0;      end // MOV
            default: begin mux_Z = arith_result; mux_C = arith_cout; end // default ADD
        endcase
    end

    assign Z = mux_Z;
    assign N_flag = mux_Z[63];
    assign Z_flag = (mux_Z == 64'd0);
    assign C_flag = mux_C;

    // V flag only meaningful for arithmetic ops
    wire is_arith = (alu_ctrl == 4'b0100) || // ADD
                    (alu_ctrl == 4'b0010) || // SUB
                    (alu_ctrl == 4'b1010) || // CMP
                    (alu_ctrl == 4'b1011);   // CMN
    assign V_flag = is_arith ? arith_V : 1'b0;

endmodule
