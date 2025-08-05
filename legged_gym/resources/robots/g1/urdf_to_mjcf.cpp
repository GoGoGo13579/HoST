#include <mujoco/mujoco.h>
#include <iostream>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <input.urdf>" << std::endl;
        return 1;
    }
    
    char error[1000] = "";
    mjModel* model = mj_loadXML(argv[1], NULL, error, 1000);
    mj_saveLastXML("g1_23dof.xml", model, error, 1000);
    std::cout << "成功转换为MJCF格式: g1_23dof.xml" << std::endl;
    mj_deleteModel(model);
    return 0;
}