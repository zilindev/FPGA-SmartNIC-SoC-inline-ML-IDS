`timescale 1ns / 1ps

// EX.v -- Execute stage: barrel shifter + ALU

module EX_stage(
    input  [63:0] R1,
    input  [63:0] R2,
    input  [63:0] sign_extended,

    input         alu_src,    // 0=shifted reg, 1=immediate
    input  [3:0]  ALU_op,
    input  [4:0]  shift_amt,
    input  [1:0]  shift_type, // 00=LSL, 01=LSR, 10=ASR, 11=ROR

    output [63:0] alu_result,
    output [63:0] write_data, // R2 pass-through for STR
    output        flag_N,
    output        flag_Z,
    output        flag_C,
    output        flag_V
);

    // Barrel shifter
    reg [63:0] shifted_R2;

    always @(*) begin
        case (shift_type)
            2'b00:   shifted_R2 = R2 << shift_amt;
            2'b01:   shifted_R2 = R2 >> shift_amt;
            2'b10:   shifted_R2 = $signed(R2) >>> shift_amt;
            2'b11:   shifted_R2 = (R2 >> shift_amt) | (R2 << (64 - shift_amt));
            default: shifted_R2 = R2;
        endcase
    end

    // Operand2 mux
    wire [63:0] operand2 = alu_src ? sign_extended : shifted_R2;

    // ALU
    alu64 u_alu (
        .A(R1),
        .B(operand2),
        .alu_ctrl(ALU_op),
        .Z(alu_result),
        .N_flag(flag_N),
        .Z_flag(flag_Z),
        .C_flag(flag_C),
        .V_flag(flag_V)
    );

    assign write_data = R2;

endmodule
