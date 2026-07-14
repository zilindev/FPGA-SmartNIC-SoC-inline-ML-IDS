// instruction_decoder.v — Combinational instruction decoder
`timescale 1ns / 1ps
`include "gpu_params.vh"

module instruction_decoder (
    input  [`INSTR_WIDTH-1:0]   instruction,
    input                       valid,

    output reg [3:0]            alu_op,
    output reg                  reg_write,
    output reg                  mem_read,
    output reg                  mem_write,
    output reg                  is_fma,
    output reg [1:0]            branch_type,
    output reg                  use_imm,
    output reg [2:0]            dtype,
    output reg                  is_halt,

    output [`REG_ADDR_W-1:0]    rd,
    output [`REG_ADDR_W-1:0]    rs1_addr,
    output [`REG_ADDR_W-1:0]    rs2_addr,
    output [`REG_ADDR_W-1:0]    rs3_addr,
    output reg [`DATA_WIDTH-1:0] imm_ext
);

    // Field extraction
    wire [4:0] opcode = instruction[`F_OPCODE_HI:`F_OPCODE_LO];

    assign rd       = instruction[`F_RD_HI:`F_RD_LO];
    assign rs1_addr = instruction[`F_RS1_HI:`F_RS1_LO];
    assign rs2_addr = instruction[`F_RS2_HI:`F_RS2_LO];
    assign rs3_addr = instruction[`F_RS3_HI:`F_RS3_LO];

    // Immediate sign-extension
    wire [`DATA_WIDTH-1:0] imm19_sext = {{45{instruction[`F_IMM19_HI]}},
                                          instruction[`F_IMM19_HI:`F_IMM19_LO]};
    wire [`DATA_WIDTH-1:0] imm12_sext = {{52{instruction[`F_OFFSET12_HI]}},
                                          instruction[`F_OFFSET12_HI:`F_OFFSET12_LO]};

    // Control signal generation
    always @(*) begin
        alu_op      = 4'd0;
        reg_write   = 1'b0;
        mem_read    = 1'b0;
        mem_write   = 1'b0;
        is_fma      = 1'b0;
        branch_type = `BR_NONE;
        use_imm     = 1'b0;
        dtype       = instruction[`F_DTYPE_HI:`F_DTYPE_LO];
        is_halt     = 1'b0;
        imm_ext     = 64'd0;

        if (valid) begin
            case (opcode)
                `OP_NOP: begin
                end

                `OP_ADD: begin
                    alu_op    = `ALU_ADD;
                    reg_write = 1'b1;
                end

                `OP_SUB: begin
                    alu_op    = `ALU_SUB;
                    reg_write = 1'b1;
                end

                `OP_MUL: begin
                    alu_op    = `ALU_MUL;
                    reg_write = 1'b1;
                end

                `OP_FMA: begin
                    alu_op    = `ALU_MUL;
                    reg_write = 1'b1;
                    is_fma    = 1'b1;
                end

                `OP_MAX: begin
                    alu_op    = `ALU_MAX;
                    reg_write = 1'b1;
                end

                `OP_MOV: begin
                    alu_op    = `ALU_MOV;
                    reg_write = 1'b1;
                end

                `OP_ADDI: begin
                    alu_op    = `ALU_ADD;
                    reg_write = 1'b1;
                    use_imm   = 1'b1;
                    imm_ext   = imm19_sext;
                end

                `OP_LD: begin
                    alu_op    = `ALU_ADD;
                    reg_write = 1'b1;
                    mem_read  = 1'b1;
                    use_imm   = 1'b1;
                    imm_ext   = imm12_sext;
                end

                `OP_ST: begin
                    alu_op    = `ALU_ADD;
                    mem_write = 1'b1;
                    use_imm   = 1'b1;
                    imm_ext   = imm12_sext;
                end

                `OP_BLT: begin
                    branch_type = `BR_BLT;
                    imm_ext     = imm19_sext;
                end

                `OP_BGE: begin
                    branch_type = `BR_BGE;
                    imm_ext     = imm19_sext;
                end

                `OP_RELU: begin
                    alu_op    = `ALU_RELU;
                    reg_write = 1'b1;
                end

                `OP_BCAST: begin
                    alu_op    = `ALU_BCAST;
                    reg_write = 1'b1;
                    use_imm   = 1'b1;
                    imm_ext   = {{62{1'b0}}, instruction[1:0]}; // lane select from func[1:0]
                end

                `OP_MOVI: begin
                    reg_write = 1'b1;
                    use_imm   = 1'b1;
                    imm_ext   = {4{instruction[15:0]}}; // broadcast to all lanes
                end

                `OP_HALT: begin
                    is_halt = 1'b1;
                end

                default: begin
                end
            endcase
        end
    end

endmodule
