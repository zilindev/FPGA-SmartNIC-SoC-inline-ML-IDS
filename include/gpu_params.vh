`ifndef GPU_PARAMS_VH
`define GPU_PARAMS_VH

//----------------------------------------------------------------------------
// General sizing
//----------------------------------------------------------------------------
// Guard DATA_WIDTH — also defined in NF2.1 registers.v (same value: 64)
`ifndef DATA_WIDTH
`define DATA_WIDTH       64          // Register / data-memory word width
`endif
`define INSTR_WIDTH      32          // Instruction width
`define REG_COUNT        16          // Number of registers (4-bit address)
`define REG_ADDR_W       4           // log2(REG_COUNT)
`define IMEM_DEPTH       1024        // Instruction memory depth (words)
`define IMEM_ADDR_W      10          // log2(IMEM_DEPTH)
`define DMEM_DEPTH       1024        // Data memory depth (words)
`define DMEM_ADDR_W      10          // log2(DMEM_DEPTH)
`define PC_WIDTH         10          // Program counter width = IMEM_ADDR_W
`define SIMD_LANES       4           // Number of SIMD lanes (64 / 16)
`define LANE_WIDTH       16          // Per-lane element width (int16 or bf16)

//----------------------------------------------------------------------------
// Instruction field positions
//----------------------------------------------------------------------------
// R-type: [31:27] opcode  [26:23] rd  [22:19] rs1  [18:15] rs2  [14:12] dtype  [11:0] func
`define F_OPCODE_HI      31
`define F_OPCODE_LO      27
`define F_RD_HI          26
`define F_RD_LO          23
`define F_RS1_HI         22
`define F_RS1_LO         19
`define F_RS2_HI         18
`define F_RS2_LO         15
`define F_DTYPE_HI       14
`define F_DTYPE_LO       12
`define F_FUNC_HI        11
`define F_FUNC_LO        0

// I-type: [31:27] opcode  [26:23] rd  [22:19] rs1  [18:0] imm19
`define F_IMM19_HI       18
`define F_IMM19_LO       0
`define IMM19_WIDTH      19

// M-type: [31:27] opcode  [26:23] rd/rs  [22:19] rbase  [18:15] unused  [14:12] dtype  [11:0] offset12
`define F_OFFSET12_HI    11
`define F_OFFSET12_LO    0
`define OFFSET12_WIDTH   12

// F-type (FMA): [31:27] opcode  [26:23] rd  [22:19] rs1  [18:15] rs2  [14:12] dtype  [11:8] rs3  [7:0] unused
`define F_RS3_HI         11
`define F_RS3_LO         8

// B-type: [31:27] opcode  [26:23] rs1  [22:19] rs2  [18:0] offset19
// (reuses IMM19 field positions for branch offset)

//----------------------------------------------------------------------------
// Opcode encoding (5 bits) — [31:27]
//----------------------------------------------------------------------------
`define OP_NOP           5'b00000    // 0x00
`define OP_ADD           5'b00001    // 0x01
`define OP_SUB           5'b00010    // 0x02
`define OP_MUL           5'b00011    // 0x03
`define OP_FMA           5'b00100    // 0x04  — Tensor core op
`define OP_MAX           5'b00101    // 0x05
`define OP_MOV           5'b01100    // 0x0C
`define OP_ADDI          5'b01110    // 0x0E
`define OP_LD            5'b10000    // 0x10
`define OP_ST            5'b10001    // 0x11
`define OP_BLT           5'b10110    // 0x16
`define OP_BGE           5'b10111    // 0x17
`define OP_RELU          5'b11000    // 0x18  — Activation function
`define OP_BCAST         5'b11010    // 0x1A  — Cross-lane broadcast
`define OP_MOVI          5'b11011    // 0x1B
`define OP_HALT          5'b11111    // 0x1F

//----------------------------------------------------------------------------
// Data type encoding (3 bits) — dtype field [14:12]
//----------------------------------------------------------------------------
`define DTYPE_S16        3'b000      // Signed 16-bit integer (4-lane SIMD)
`define DTYPE_U16        3'b001      // Unsigned 16-bit integer
`define DTYPE_S32        3'b010      // Signed 32-bit (2-lane or scalar)
`define DTYPE_BF16       3'b100      // BFloat16 (4-lane SIMD)

//----------------------------------------------------------------------------
// ALU operation select (internal control, not ISA-visible)
//----------------------------------------------------------------------------
`define ALU_ADD          4'd0
`define ALU_SUB          4'd1
`define ALU_MUL          4'd2
`define ALU_MAX          4'd3
`define ALU_MOV          4'd4
`define ALU_RELU         4'd5
`define ALU_BCAST        4'd6

//----------------------------------------------------------------------------
// Branch type encoding (internal control)
//----------------------------------------------------------------------------
`define BR_NONE          2'b00
`define BR_BLT           2'b01
`define BR_BGE           2'b10

//----------------------------------------------------------------------------
// Special register indices
//----------------------------------------------------------------------------
`define REG_ZERO         4'd0        // Hardwired zero
`define REG_THREADID     4'd13       // threadIdx.x
`define REG_BLOCKID      4'd14       // blockIdx.x
`define REG_BLOCKDIM     4'd15       // blockDim.x

//----------------------------------------------------------------------------
// BFloat16 constants
//----------------------------------------------------------------------------
`define BF16_WIDTH       16
`define BF16_EXP_W       8
`define BF16_MANT_W      7
`define BF16_BIAS        8'd127
`define BF16_ZERO_POS    16'h0000
`define BF16_ZERO_NEG    16'h8000
`define BF16_INF_POS     16'h7F80
`define BF16_INF_NEG     16'hFF80
`define BF16_NAN         16'h7FC0    // Canonical quiet NaN
`define BF16_ONE         16'h3F80    // 1.0 in BF16

`endif // GPU_PARAMS_VH
