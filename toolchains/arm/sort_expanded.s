	.cpu arm7tdmi
	.arch armv4t
	.fpu softvfp
	.eabi_attribute 20, 1
	.eabi_attribute 21, 1
	.eabi_attribute 23, 3
	.eabi_attribute 24, 1
	.eabi_attribute 25, 1
	.eabi_attribute 26, 1
	.eabi_attribute 30, 6
	.eabi_attribute 34, 0
	.eabi_attribute 18, 4
	.file	"sort.c"
	.text
	.section	.rodata
	.align	2
.LC0:
	.word	323
	.word	123
	.word	-455
	.word	2
	.word	98
	.word	125
	.word	10
	.word	65
	.word	-56
	.word	0
	.text
	.align	2
	.global	main
	.syntax unified
	.arm
	.type	main, %function
main:
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 56
	@ frame_needed = 1, uses_anonymous_args = 0
	@ --- PUSH {fp, lr} expanded ---
	sub	sp, sp, #8
	str	fp, [sp, #0]
	str	lr, [sp, #4]
	add	fp, sp, #4
	sub	sp, sp, #56
	ldr	r3, .L8
	sub	ip, fp, #56
	mov	lr, r3
	@ --- LDMIA lr!, {r0, r1, r2, r3} expanded ---
	ldr	r0, [lr, #0]
	ldr	r1, [lr, #4]
	ldr	r2, [lr, #8]
	ldr	r3, [lr, #12]
	add	lr, lr, #16
	@ --- STMIA ip!, {r0, r1, r2, r3} expanded ---
	str	r0, [ip, #0]
	str	r1, [ip, #4]
	str	r2, [ip, #8]
	str	r3, [ip, #12]
	add	ip, ip, #16
	@ --- LDMIA lr!, {r0, r1, r2, r3} expanded ---
	ldr	r0, [lr, #0]
	ldr	r1, [lr, #4]
	ldr	r2, [lr, #8]
	ldr	r3, [lr, #12]
	add	lr, lr, #16
	@ --- STMIA ip!, {r0, r1, r2, r3} expanded ---
	str	r0, [ip, #0]
	str	r1, [ip, #4]
	str	r2, [ip, #8]
	str	r3, [ip, #12]
	add	ip, ip, #16
	@ --- LDMIA lr, {r0, r1} expanded ---
	ldr	r0, [lr, #0]
	ldr	r1, [lr, #4]
	@ --- STMIA ip, {r0, r1} expanded ---
	str	r0, [ip, #0]
	str	r1, [ip, #4]
	mov	r3, #0
	str	r3, [fp, #-8]
	b	.L2
.L6:
	ldr	r3, [fp, #-8]
	add	r3, r3, #1
	str	r3, [fp, #-12]
	b	.L3
.L5:
	ldr	r3, [fp, #-12]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r2, [r3, #-52]
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r3, [r3, #-52]
	cmp	r2, r3
	bge	.L4
	ldr	r3, [fp, #-12]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r3, [r3, #-52]
	str	r3, [fp, #-16]
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r2, [r3, #-52]
	ldr	r3, [fp, #-12]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	str	r2, [r3, #-52]
	ldr	r3, [fp, #-8]
	lsl	r3, r3, #2
	sub	r3, r3, #4
	add	r3, r3, fp
	ldr	r2, [fp, #-16]
	str	r2, [r3, #-52]
.L4:
	ldr	r3, [fp, #-12]
	add	r3, r3, #1
	str	r3, [fp, #-12]
.L3:
	ldr	r3, [fp, #-12]
	cmp	r3, #9
	ble	.L5
	ldr	r3, [fp, #-8]
	add	r3, r3, #1
	str	r3, [fp, #-8]
.L2:
	ldr	r3, [fp, #-8]
	cmp	r3, #9
	ble	.L6
	mov	r3, #0
	mov	r0, r3
	sub	sp, fp, #4
	@ sp needed
	@ --- POP {fp, lr} expanded ---
	ldr	fp, [sp, #0]
	ldr	lr, [sp, #4]
	add	sp, sp, #8
	bx	lr
.L9:
	.align	2
.L8:
	.word	.LC0
	.size	main, .-main
	.ident	"GCC: (Arm GNU Toolchain 15.2.Rel1 (Build arm-15.86)) 15.2.1 20251203"
