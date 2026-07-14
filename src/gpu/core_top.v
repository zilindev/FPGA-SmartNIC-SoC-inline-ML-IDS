// core_top.v
`timescale 1ns / 1ps
`include "gpu_params.vh"

module core_top (
    input         clk,
    input         rst,

    input  [31:0] thread_id,
    input  [31:0] block_id,
    input  [31:0] block_dim,

    output        halted,

    // Instruction Memory interface
    output [`IMEM_ADDR_W-1:0] imem_addr,
    output                    imem_rd_en,
    input  [`INSTR_WIDTH-1:0] imem_data,

    // Data Memory interface
    output [`DMEM_ADDR_W-1:0] dmem_addr,
    output [`DATA_WIDTH-1:0]  dmem_wr_data,
    output                    dmem_wr_en,
    input  [`DATA_WIDTH-1:0]  dmem_rd_data
);

    // Pipeline control
    wire stall;
    wire ex1_branch_taken;

    // Hold IMEM output during stall, halt, or branch flush
    assign imem_rd_en = !stall && !halted && !ex1_branch_taken;

    // IF/ID outputs
    wire [`INSTR_WIDTH-1:0] ifid_instruction;
    wire [`PC_WIDTH-1:0]    ifid_pc;
    wire                    ifid_valid;

    // Decode stage outputs
    wire [3:0]  dec_alu_op;
    wire        dec_reg_write;
    wire        dec_mem_read;
    wire        dec_mem_write;
    wire        dec_is_fma;
    wire        dec_use_imm;
    wire [2:0]  dec_dtype;
    wire        dec_is_halt;
    wire [1:0]  dec_branch_type;

    wire [`REG_ADDR_W-1:0]  dec_rd;
    wire [`REG_ADDR_W-1:0]  dec_rs1_addr;
    wire [`REG_ADDR_W-1:0]  dec_rs2_addr;
    wire [`REG_ADDR_W-1:0]  dec_rs3_addr;
    wire [`DATA_WIDTH-1:0]  dec_rs1_data;
    wire [`DATA_WIDTH-1:0]  dec_rs2_data;
    wire [`DATA_WIDTH-1:0]  dec_rs3_data;
    wire [`DATA_WIDTH-1:0]  dec_imm_ext;
    wire [`PC_WIDTH-1:0]    dec_pc;

    // ID/EX outputs
    wire [3:0]  idex_alu_op;
    wire        idex_reg_write;
    wire        idex_mem_read;
    wire        idex_mem_write;
    wire        idex_is_fma;
    wire        idex_use_imm;
    wire [2:0]  idex_dtype;
    wire        idex_is_halt;
    wire [1:0]  idex_branch_type;

    wire [`REG_ADDR_W-1:0]  idex_rd;
    wire [`DATA_WIDTH-1:0]  idex_rs1_data;
    wire [`DATA_WIDTH-1:0]  idex_rs2_data;
    wire [`DATA_WIDTH-1:0]  idex_rs3_data;
    wire [`DATA_WIDTH-1:0]  idex_imm_ext;
    wire [`PC_WIDTH-1:0]    idex_pc;

    // EX1: Branch resolution
    wire signed_lt = ($signed(idex_rs1_data) < $signed(idex_rs2_data));

    assign ex1_branch_taken = (idex_branch_type == `BR_BLT &&  signed_lt) ||
                              (idex_branch_type == `BR_BGE && !signed_lt);

    wire [`PC_WIDTH-1:0] ex1_branch_target = idex_pc + idex_imm_ext[`PC_WIDTH-1:0];

    // Execute stage outputs
    wire [`DATA_WIDTH-1:0]  ex_result;
    wire                    dmem_wr_en_raw;

    // EX1/EX2 outputs
    wire        ex1ex2_reg_write;
    wire        ex1ex2_mem_read;
    wire [`REG_ADDR_W-1:0]  ex1ex2_rd;
    wire                    ex1ex2_is_halt;

    // EX2/EX3 outputs
    wire        ex2ex3_reg_write;
    wire        ex2ex3_mem_read;
    wire [`REG_ADDR_W-1:0]  ex2ex3_rd;
    wire                    ex2ex3_is_halt;

    // EX3/WB outputs
    wire        exwb_reg_write;
    wire        exwb_mem_read;
    wire [`REG_ADDR_W-1:0]  exwb_rd;
    wire [`DATA_WIDTH-1:0]  exwb_alu_result;
    wire                    exwb_is_halt;

    // Writeback outputs
    wire                    wb_reg_write;
    wire [`REG_ADDR_W-1:0]  wb_rd;
    wire [`DATA_WIDTH-1:0]  wb_data;

    // Program Counter
    program_counter u_pc (
        .clk           (clk),
        .rst           (rst),
        .stall         (stall || halted),
        .branch_en     (ex1_branch_taken),
        .branch_target (ex1_branch_target),
        .pc_out        (imem_addr)
    );

    // Fetch Stage
    assign ifid_instruction = imem_data;

    fetch_stage u_fetch (
        .clk       (clk),
        .rst       (rst),
        .stall     (stall),
        .flush     (ex1_branch_taken),
        .pc_in     (imem_addr),
        .pc_out    (ifid_pc),
        .valid_out (ifid_valid)
    );

    // Decode Stage
    decode_stage u_decode (
        .clk             (clk),
        .rst             (rst),
        .stall           (stall),
        .flush           (ex1_branch_taken),
        .instruction     (ifid_instruction),
        .pc_in           (ifid_pc),
        .valid_in        (ifid_valid),
        .thread_id       (thread_id),
        .block_id        (block_id),
        .block_dim       (block_dim),
        .wb_reg_write    (wb_reg_write),
        .wb_rd           (wb_rd),
        .wb_data         (wb_data),
        .alu_op_out      (dec_alu_op),
        .reg_write_out   (dec_reg_write),
        .mem_read_out    (dec_mem_read),
        .mem_write_out   (dec_mem_write),
        .is_fma_out      (dec_is_fma),
        .use_imm_out     (dec_use_imm),
        .dtype_out       (dec_dtype),
        .is_halt_out     (dec_is_halt),
        .branch_type_out (dec_branch_type),
        .rd_out          (dec_rd),
        .rs1_addr_out    (dec_rs1_addr),
        .rs2_addr_out    (dec_rs2_addr),
        .rs3_addr_out    (dec_rs3_addr),
        .rs1_data_out    (dec_rs1_data),
        .rs2_data_out    (dec_rs2_data),
        .rs3_data_out    (dec_rs3_data),
        .imm_ext_out     (dec_imm_ext),
        .pc_out          (dec_pc)
    );

    // ID/EX Pipeline Register (flush on stall or branch)
    pipeline_reg_idex u_idex (
        .clk             (clk),
        .rst             (rst),
        .stall           (stall),
        .flush           (stall || ex1_branch_taken),
        .alu_op_in       (dec_alu_op),
        .reg_write_in    (dec_reg_write),
        .mem_read_in     (dec_mem_read),
        .mem_write_in    (dec_mem_write),
        .is_fma_in       (dec_is_fma),
        .use_imm_in      (dec_use_imm),
        .dtype_in        (dec_dtype),
        .is_halt_in      (dec_is_halt),
        .branch_type_in  (dec_branch_type),
        .rd_in           (dec_rd),
        .rs1_data_in     (dec_rs1_data),
        .rs2_data_in     (dec_rs2_data),
        .rs3_data_in     (dec_rs3_data),
        .imm_ext_in      (dec_imm_ext),
        .pc_in           (dec_pc),
        .alu_op_out      (idex_alu_op),
        .reg_write_out   (idex_reg_write),
        .mem_read_out    (idex_mem_read),
        .mem_write_out   (idex_mem_write),
        .is_fma_out      (idex_is_fma),
        .use_imm_out     (idex_use_imm),
        .dtype_out       (idex_dtype),
        .is_halt_out     (idex_is_halt),
        .branch_type_out (idex_branch_type),
        .rd_out          (idex_rd),
        .rs1_data_out    (idex_rs1_data),
        .rs2_data_out    (idex_rs2_data),
        .rs3_data_out    (idex_rs3_data),
        .imm_ext_out     (idex_imm_ext),
        .pc_out          (idex_pc)
    );

    // Execute Stage
    execute_stage u_execute (
        .clk           (clk),
        .rst           (rst),
        .alu_op        (idex_alu_op),
        .is_fma        (idex_is_fma),
        .use_imm       (idex_use_imm),
        .dtype         (idex_dtype),
        .mem_write     (idex_mem_write),
        .rs1_data      (idex_rs1_data),
        .rs2_data      (idex_rs2_data),
        .rs3_data      (idex_rs3_data),
        .imm_ext       (idex_imm_ext),
        .result        (ex_result),
        .dmem_addr     (dmem_addr),
        .dmem_wr_data  (dmem_wr_data),
        .dmem_wr_en    (dmem_wr_en_raw)
    );

    // Gate DMEM writes after HALT
    assign dmem_wr_en = dmem_wr_en_raw && !halted;

    // EX1/EX2 Pipeline Register (control only)
    pipeline_reg_ex u_ex1ex2 (
        .clk            (clk),
        .rst            (rst),
        .flush          (1'b0),
        .reg_write_in   (idex_reg_write),
        .mem_read_in    (idex_mem_read),
        .rd_in          (idex_rd),
        .is_halt_in     (idex_is_halt),
        .reg_write_out  (ex1ex2_reg_write),
        .mem_read_out   (ex1ex2_mem_read),
        .rd_out         (ex1ex2_rd),
        .is_halt_out    (ex1ex2_is_halt)
    );

    // EX2/EX3 Pipeline Register
    pipeline_reg_ex u_ex2ex3 (
        .clk            (clk),
        .rst            (rst),
        .flush          (1'b0),
        .reg_write_in   (ex1ex2_reg_write),
        .mem_read_in    (ex1ex2_mem_read),
        .rd_in          (ex1ex2_rd),
        .is_halt_in     (ex1ex2_is_halt),
        .reg_write_out  (ex2ex3_reg_write),
        .mem_read_out   (ex2ex3_mem_read),
        .rd_out         (ex2ex3_rd),
        .is_halt_out    (ex2ex3_is_halt)
    );

    // EX3/WB Pipeline Register
    pipeline_reg_exwb u_exwb (
        .clk            (clk),
        .rst            (rst),
        .reg_write_in   (ex2ex3_reg_write),
        .mem_read_in    (ex2ex3_mem_read),
        .rd_in          (ex2ex3_rd),
        .alu_result_in  (ex_result),
        .is_halt_in     (ex2ex3_is_halt),
        .reg_write_out  (exwb_reg_write),
        .mem_read_out   (exwb_mem_read),
        .rd_out         (exwb_rd),
        .alu_result_out (exwb_alu_result),
        .is_halt_out    (exwb_is_halt)
    );

    // Writeback Stage
    writeback_stage u_writeback (
        .reg_write_in  (exwb_reg_write),
        .mem_read      (exwb_mem_read),
        .rd_in         (exwb_rd),
        .alu_result    (exwb_alu_result),
        .mem_data      (dmem_rd_data),
        .reg_write_out (wb_reg_write),
        .rd_out        (wb_rd),
        .data_out      (wb_data)
    );

    // Hazard Unit (stall-only, no forwarding)
    hazard_unit u_hazard (
        .de_rs1        (dec_rs1_addr),
        .de_rs2        (dec_rs2_addr),
        .de_rs3        (dec_rs3_addr),
        .de_is_fma     (dec_is_fma),
        .ex1_rd        (idex_rd),
        .ex1_reg_write (idex_reg_write),
        .ex2_rd        (ex1ex2_rd),
        .ex2_reg_write (ex1ex2_reg_write),
        .ex3_rd        (ex2ex3_rd),
        .ex3_reg_write (ex2ex3_reg_write),
        .wb_rd         (exwb_rd),
        .wb_reg_write  (exwb_reg_write),
        .stall         (stall)
    );

    // Halt detection (registered to avoid timing path through dmem_wr_en)
    reg halted_r;
    always @(posedge clk) begin
        if (rst)
            halted_r <= 1'b0;
        else if (exwb_is_halt)
            halted_r <= 1'b1;
    end
    assign halted = halted_r;

endmodule
