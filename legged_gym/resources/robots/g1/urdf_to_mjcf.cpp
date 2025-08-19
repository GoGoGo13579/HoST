// 简单的 URDF -> MJCF 转换器（基于 MuJoCo 自带的 URDF 解析能力）
// 用法： urdf_to_mjcf <input.urdf> [output.xml]

#include <mujoco/mujoco.h>
#include <iostream>
#include <string>
#include <cstring>

static std::string deriveOutputPathFromInput(const std::string& inputPath) {
    // 将 .urdf 或 .xml 后缀改为 _converted.xml；否则追加 .xml
    const std::string::size_type dot = inputPath.find_last_of('.') ;
    if (dot != std::string::npos) {
        const std::string ext = inputPath.substr(dot);
        if (ext == ".urdf" || ext == ".URDF" || ext == ".xml" || ext == ".XML") {
            return inputPath.substr(0, dot) + "_converted.xml";
        }
    }
    return inputPath + ".xml";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <input.urdf> [output.xml]" << std::endl;
        std::cerr << "说明: 依赖 MuJoCo 对 URDF 的解析（URDF 中可包含 <mujoco><compiler .../> 指令）。" << std::endl;
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = (argc >= 3) ? std::string(argv[2]) : deriveOutputPathFromInput(inputPath);

    char error[1024] = {0};
    // 直接从文件系统加载。若需要自定义 VFS，可在此构造 mjVFS。
    mjModel* model = mj_loadXML(inputPath.c_str(), nullptr, error, sizeof(error));
    if (!model) {
        std::cerr << "加载失败: " << inputPath << std::endl;
        if (std::strlen(error) > 0) {
            std::cerr << "错误信息: " << error << std::endl;
        }
        return 2;
    }

    if (!mj_saveLastXML(outputPath.c_str(), model, error, sizeof(error))) {
        std::cerr << "保存 MJCF 失败: " << outputPath << std::endl;
        if (std::strlen(error) > 0) {
            std::cerr << "错误信息: " << error << std::endl;
        }
        mj_deleteModel(model);
        return 3;
    }

    std::cout << "成功转换为 MJCF: " << outputPath << std::endl;
    mj_deleteModel(model);
    return 0;
}