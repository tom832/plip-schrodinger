import sys
import os
import numpy as np
from scipy.spatial import distance

from schrodinger import structure
from schrodinger.structutils import analyze, assignbondorders, build
from schrodinger.structutils import interactions
from schrodinger.structutils.interactions import hbond, salt_bridge, pi
from schrodinger.application.livedesign import lid
from schrodinger.utils import qapplication

# 导入 ProtAssign
try:
    from schrodinger.protein import assignment
except ImportError:
    print("错误: 无法导入 schrodinger.protein.assignment 模块。")
    sys.exit(1)

def get_atom_str(atom):
    return f"{atom.pdbres.strip()}:{atom.resnum}:{atom.atom_name.strip()}"

def clean_structure(st):
    """清洗结构：删除水分子"""
    print(">>> [清洗] 正在移除溶剂和水分子...")
    water_atoms = analyze.evaluate_asl(st, "water")
    if water_atoms:
        st.deleteAtoms(water_atoms)
        print(f"    已删除 {len(water_atoms)} 个水原子。")

def preprocess_structure(st):
    """预处理：分配键级 -> 加氢 -> 优化氢键网络"""
    print(">>> [预处理] 正在分配键级...")
    try:
        assignbondorders.assign_st(st)
    except Exception as e:
        print(f"    警告: 键级分配可能不完全: {e}")

    print(">>> [预处理] 添加氢原子...")
    build.add_hydrogens(st)

    print(">>> [预处理] 正在运行 ProtAssign 优化氢键网络...")
    try:
        prot_assign = assignment.ProtAssign(st, asl="all", interactive=False, sample_waters=False)
        print(f"    优化完成。当前结构原子数: {st.atom_total}")
    except Exception as e:
        print(f"    警告: 氢键网络优化步骤发生异常 (可忽略): {e}")

def split_ligand_receptor_smart(st):
    """智能分离配体和受体"""
    # 1. 提取受体
    receptor_asl = "(protein) or (nucleic_acids)"
    receptor_indices = analyze.evaluate_asl(st, receptor_asl)
    if not receptor_indices:
        receptor_indices = analyze.evaluate_asl(st, "chain.atom_total > 500")
    if not receptor_indices:
        raise ValueError("无法识别受体。")

    rec_st = st.extract(receptor_indices)
    rec_st.title = "Receptor"

    # 2. 智能查找配体
    print("    正在扫描所有潜在配体...")
    ligands = analyze.find_ligands(st)
    candidates = []
    
    for lig in ligands:
        try:
            indices = lig.atom_indexes
        except AttributeError:
            indices = lig.atom_indices
        
        if any(idx in receptor_indices for idx in indices): continue
        if len(indices) < 6: continue
            
        first_atom = st.atom[indices[0]]
        res_name = first_atom.pdbres.strip()
        candidates.append({
            "indices": indices,
            "count": len(indices),
            "name": res_name,
            "resnum": first_atom.resnum
        })

    if not candidates:
        raise ValueError("未找到符合条件(原子数>6)的有机配体分子。")

    candidates.sort(key=lambda x: x["count"], reverse=True)
    best_ligand = candidates[0]
    print(f"    >>> 自动选择最大的配体: {best_ligand['name']}:{best_ligand['resnum']}")
    
    lig_st = st.extract(best_ligand["indices"])
    lig_st.title = f"Ligand_{best_ligand['name']}"
    
    return lig_st, rec_st

def keep_closest_chain(rec_st, lig_st):
    """
    [V11 修复版] 使用 Numpy 切片获取坐标，兼容所有 API 版本。
    """
    print("\n>>> [筛选] 正在寻找与配体结合的特定蛋白链...")
    
    best_chain_name = None
    min_dist = float('inf')
    
    # 1. 获取坐标矩阵
    # getXYZ() 返回的是 numpy array，不接受参数
    lig_coords = lig_st.getXYZ()
    all_rec_coords = rec_st.getXYZ() # 获取整个受体的坐标
    
    # 2. 遍历每一条链
    for chain in rec_st.chain:
        # 获取 Schrödinger 的原子索引 (从1开始)
        schrodinger_indices = chain.getAtomIndices()
        if not schrodinger_indices:
            continue
            
        # 转换为 NumPy 的索引 (从0开始)
        # 这一步至关重要：Python 列表索引 = Schrödinger ID - 1
        numpy_indices = [i - 1 for i in schrodinger_indices]
        
        # 使用 NumPy 切片提取当前链的坐标
        chain_coords = all_rec_coords[numpy_indices]
        
        # 3. 计算距离
        dists = distance.cdist(lig_coords, chain_coords)
        current_min_dist = np.min(dists)
        
        chain_id = chain.name if chain.name.strip() else "_"
        
        if current_min_dist < min_dist:
            min_dist = current_min_dist
            best_chain_name = chain.name

    print(f"    >>> 锁定目标链: Chain '{best_chain_name}' (最短距离: {min_dist:.2f} Å)")
    
    if best_chain_name.strip() == "":
        asl = "chain ' ' "
    else:
        asl = f"chain {best_chain_name}"
        
    indices = analyze.evaluate_asl(rec_st, asl)
    single_chain_rec = rec_st.extract(indices)
    single_chain_rec.title = f"Receptor_Chain_{best_chain_name}"
    
    return single_chain_rec

def save_compatible_structures(rec_st, lig_st, base_name):
    """保存 PyMOL 兼容结构"""
    print("\n>>> 正在生成 PyMOL 兼容的结构文件...")
    try:
        merged_st = rec_st.copy()
        lig_copy = lig_st.copy()
        
        # 强制修改配体链名为 Z
        for atom in lig_copy.atom:
            atom.chain = "Z"
        
        # 使用 extend 合并
        merged_st.extend(lig_copy)
        merged_st.title = f"{base_name}_Complex_Clean"

        output_pdb = base_name + "_pymol_ready.pdb"
        merged_st.write(output_pdb)
        print(f"  [PDB] PyMOL 专用文件(单链)已保存至 -> {output_pdb}")

        output_mae = base_name + "_optimized.mae"
        merged_st.write(output_mae)
        print(f"  [MAE] 薛定谔格式已保存至 -> {output_mae}")

        output_sdf = base_name + "_ligand.sdf"
        lig_copy.write(output_sdf)
        print(f"  [SDF] 配体单独文件已保存至 -> {output_sdf}")

    except Exception as e:
        print(f"保存结构失败: {e}")
        import traceback
        traceback.print_exc()

def run_analysis(input_file):
    print(f"--- 正在处理: {input_file} ---")
    
    try:
        complex_st = structure.Structure.read(input_file)
    except Exception as e:
        print(f"错误: 无法读取文件. {e}")
        return

    clean_structure(complex_st)
    preprocess_structure(complex_st)

    try:
        # 1. 初步拆分
        lig_st, full_rec_st = split_ligand_receptor_smart(complex_st)
        
        # 2. 过滤最近链
        rec_st = keep_closest_chain(full_rec_st, lig_st)
        
        print(f"最终组装: 配体 ({lig_st.atom_total} atoms) / 单链受体 ({rec_st.atom_total} atoms)")
        
    except ValueError as e:
        print(f"拆分/筛选失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n>>> 开始相互作用分析...")
    # 简化的调用
    hbonds = hbond.get_hydrogen_bonds(lig_st, st2=rec_st)
    xbonds = hbond.get_halogen_bonds(lig_st, st2=rec_st)
    salt_bridges = salt_bridge.get_salt_bridges(lig_st, struc2=rec_st)
    pipis = pi.find_pi_pi_interactions(lig_st, struct2=rec_st)
    picats = pi.find_pi_cation_interactions(lig_st, struct2=rec_st)
    
    print(f"  摘要: 氢键 {len(hbonds)}, 卤键 {len(xbonds)}, 盐桥 {len(salt_bridges)}, Pi-Pi {len(pipis)}, Pi-Cat {len(picats)}")

    # 渲染示意图
    print("\n>>> 正在生成相互作用示意图 (LID)...")
    try:
        app = qapplication.get_application()
        image = lid.generate_lid(lig_st, rec_st)
        if image and not image.isNull():
            output_img_name = os.path.splitext(input_file)[0] + "_LID.png"
            image.save(output_img_name)
            print(f"成功: 示意图已保存至 -> {output_img_name}")
    except Exception as e:
        print(f"渲染示意图失败: {e}")

    # 保存结构
    base_name = os.path.splitext(input_file)[0]
    save_compatible_structures(rec_st, lig_st, base_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: $SCHRODINGER/run python3 analyze_interactions_v11.py <complex.pdb>")
    else:
        run_analysis(sys.argv[1])