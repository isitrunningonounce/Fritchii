// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <optional>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/object_pool.h"

namespace {
using namespace Shader;

class HalfEnvironment final : public Environment {
public:
    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    TextureType ReadTextureType(u32) override { return TextureType::Color2D; }
    TexturePixelFormat ReadTexturePixelFormat(u32) override { return TexturePixelFormat::R32_UINT; }
    bool IsTexturePixelFormatInteger(u32) override { return true; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return std::nullopt; }
    void Dump(u64, u64) override {}
};

const IR::Inst* Source(const IR::Inst* inst, IR::Opcode opcode) {
    if (inst == nullptr || inst->GetOpcode() != opcode || inst->Arg(0).IsImmediate()) {
        return nullptr;
    }
    return inst->Arg(0).InstRecursive();
}

// The single-precision value stored at a shared memory offset, below its bit cast.
const IR::Inst* StoredFloat(IR::Block& block, u32 offset) {
    for (const IR::Inst& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::WriteSharedU32 && inst.Arg(0).IsImmediate() &&
            inst.Arg(0).U32() == offset && !inst.Arg(1).IsImmediate()) {
            return Source(inst.Arg(1).InstRecursive(), IR::Opcode::BitCastU32F32);
        }
    }
    return nullptr;
}
} // Anonymous namespace

TEST_CASE("Halves lowered to single precision round computed results when widened",
          "[shader_recompiler][fp16]") {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::IREmitter ir{block};
    const IR::U32 packed{ir.LoadShared(32, false, ir.Imm32(0))};
    const IR::Value halves{ir.UnpackFloat2x16(packed)};
    const IR::F16 x{ir.CompositeExtract(halves, 0)};
    const IR::F16 y{ir.CompositeExtract(halves, 1)};
    const IR::F32 sum{ir.FPConvert(32, ir.FPAdd(x, y))};
    const IR::F32 input{ir.FPConvert(32, x)};
    ir.WriteShared(32, ir.Imm32(4), ir.BitCast<IR::U32>(sum));
    ir.WriteShared(32, ir.Imm32(8), ir.BitCast<IR::U32>(input));

    IR::Program program;
    program.blocks.push_back(&block);
    program.post_order_blocks.push_back(&block);
    HalfEnvironment env;
    Optimization::LowerFp16ToFp32(program);
    Optimization::ConstantPropagationPass(env, program);

    // The single-precision sum goes through a half pack, which the unpack must not fold away.
    const IR::Inst* const rounded{StoredFloat(block, 4)};
    const IR::Inst* const unpacked{Source(rounded, IR::Opcode::CompositeExtractF32x2)};
    const IR::Inst* const repacked{Source(unpacked, IR::Opcode::UnpackHalf2x16)};
    const IR::Inst* const pair{Source(repacked, IR::Opcode::PackHalf2x16)};
    REQUIRE(pair != nullptr);
    REQUIRE(pair->GetOpcode() == IR::Opcode::CompositeConstructF32x2);
    REQUIRE(!pair->Arg(0).IsImmediate());
    REQUIRE(pair->Arg(0).InstRecursive()->GetOpcode() == IR::Opcode::FPAdd32);

    // A half read from a packed register is already representable and is widened as it is.
    const IR::Inst* const widened{StoredFloat(block, 8)};
    const IR::Inst* const read{Source(widened, IR::Opcode::CompositeExtractF32x2)};
    const IR::Inst* const loaded{Source(read, IR::Opcode::UnpackHalf2x16)};
    REQUIRE(loaded != nullptr);
    REQUIRE(loaded->GetOpcode() == IR::Opcode::LoadSharedU32);
}
