from ScriptParser import *
from Timer import *
from ScriptConverter import *

import pandas as pd
from sklearn.linear_model import LinearRegression

features = [
    'cycles',
    'branch-instructions',
    'branch-misses',
    'bus-cycles',
    'cache-misses',
    'cache-references',
    'cpu-cycles',
    'instructions',
    'mem-loads',
    'mem-stores',
    'ref-cycles',
    'topdown-fetch-bubbles',
    'topdown-recovery-bubbles',
    'topdown-slots-issued',
    'topdown-slots-retired',
    'topdown-total-slots',
    'l1d.replacement',
    'l1d_pend_miss.fb_full',
    'l1d_pend_miss.pending',
    'l1d_pend_miss.pending_cycles',
    'l1d_pend_miss.pending_cycles_any',
    'l1d_pend_miss.request_fb_full',
    'l2_demand_rqsts.wb_hit',
    'l2_lines_in.all',
    'l2_lines_in.e',
    'l2_lines_in.i',
    'l2_lines_in.s',
    'l2_lines_out.demand_clean',
    'l2_lines_out.demand_dirty',
    'l2_rqsts.all_code_rd',
    'l2_rqsts.all_demand_data_rd',
    'l2_rqsts.all_demand_miss',
    'l2_rqsts.all_demand_references',
    'l2_rqsts.all_pf',
    'l2_rqsts.all_rfo',
    'l2_rqsts.code_rd_hit',
    'l2_rqsts.code_rd_miss',
    'l2_rqsts.demand_data_rd_hit',
    'l2_rqsts.demand_data_rd_miss',
    'l2_rqsts.l2_pf_hit',
    'l2_rqsts.l2_pf_miss',
    'l2_rqsts.miss',
    'l2_rqsts.references',
    'l2_rqsts.rfo_hit',
    'l2_rqsts.rfo_miss',
    'l2_trans.all_pf',
    'l2_trans.all_requests',
    'l2_trans.code_rd',
    'l2_trans.demand_data_rd',
    'l2_trans.l1d_wb',
    'l2_trans.l2_fill',
    'l2_trans.l2_wb',
    'l2_trans.rfo',
    'lock_cycles.cache_lock_duration',
    'longest_lat_cache.miss',
    'longest_lat_cache.reference',
    'mem_load_uops_l3_hit_retired.xsnp_hit',
    'mem_load_uops_l3_hit_retired.xsnp_hitm',
    'mem_load_uops_l3_hit_retired.xsnp_miss',
    'mem_load_uops_l3_hit_retired.xsnp_none',
    'mem_load_uops_l3_miss_retired.local_dram',
    'mem_load_uops_retired.hit_lfb',
    'mem_load_uops_retired.l1_hit',
    'mem_load_uops_retired.l1_miss',
    'mem_load_uops_retired.l2_hit',
    'mem_load_uops_retired.l2_miss',
    'mem_load_uops_retired.l3_hit',
    'mem_load_uops_retired.l3_miss',
    'mem_uops_retired.all_loads',
    'mem_uops_retired.all_stores',
    'mem_uops_retired.lock_loads',
    'mem_uops_retired.split_loads',
    'mem_uops_retired.split_stores',
    'mem_uops_retired.stlb_miss_loads',
    'mem_uops_retired.stlb_miss_stores',
    'offcore_requests.all_data_rd',
    'offcore_requests.demand_code_rd',
    'offcore_requests.demand_data_rd',
    'offcore_requests.demand_rfo',
    'offcore_requests_buffer.sq_full',
    'offcore_requests_outstanding.all_data_rd',
    'offcore_requests_outstanding.cycles_with_data_rd',
    'offcore_requests_outstanding.cycles_with_demand_data_rd',
    'offcore_requests_outstanding.cycles_with_demand_rfo',
    'offcore_requests_outstanding.demand_code_rd',
    'offcore_requests_outstanding.demand_data_rd',
    'offcore_requests_outstanding.demand_data_rd_ge_6',
    'offcore_requests_outstanding.demand_rfo',
    'offcore_response',
    'offcore_response.all_code_rd.l3_hit.hit_other_core_no_fwd',
    'offcore_response.all_data_rd.l3_hit.hit_other_core_no_fwd',
    'offcore_response.all_data_rd.l3_hit.hitm_other_core',
    'offcore_response.all_reads.l3_hit.hit_other_core_no_fwd',
    'offcore_response.all_reads.l3_hit.hitm_other_core',
    'offcore_response.all_requests.l3_hit.any_response',
    'offcore_response.all_rfo.l3_hit.hit_other_core_no_fwd',
    'offcore_response.all_rfo.l3_hit.hitm_other_core',
    'offcore_response.demand_code_rd.l3_hit.hit_other_core_no_fwd',
    'offcore_response.demand_code_rd.l3_hit.hitm_other_core',
    'offcore_response.demand_data_rd.l3_hit.hit_other_core_no_fwd',
    'offcore_response.demand_data_rd.l3_hit.hitm_other_core',
    'offcore_response.demand_rfo.l3_hit.hit_other_core_no_fwd',
    'offcore_response.demand_rfo.l3_hit.hitm_other_core',
    'offcore_response.pf_l2_code_rd.l3_hit.any_response',
    'offcore_response.pf_l2_data_rd.l3_hit.any_response',
    'offcore_response.pf_l2_rfo.l3_hit.any_response',
    'offcore_response.pf_l3_code_rd.l3_hit.any_response',
    'offcore_response.pf_l3_data_rd.l3_hit.any_response',
    'offcore_response.pf_l3_rfo.l3_hit.any_response',
    'sq_misc.split_lock',
    'avx_insts.all',
    'fp_assist.any',
    'fp_assist.simd_input',
    'fp_assist.simd_output',
    'fp_assist.x87_input',
    'fp_assist.x87_output',
    'other_assists.avx_to_sse',
    'other_assists.sse_to_avx',
    'dsb2mite_switches.penalty_cycles',
    'icache.hit',
    'icache.ifdata_stall',
    'icache.ifetch_stall',
    'icache.misses',
    'idq.all_dsb_cycles_4_uops',
    'idq.all_dsb_cycles_any_uops',
    'idq.all_mite_cycles_4_uops',
    'idq.all_mite_cycles_any_uops',
    'idq.dsb_cycles',
    'idq.dsb_uops',
    'idq.empty',
    'idq.mite_all_uops',
    'idq.mite_cycles',
    'idq.mite_uops',
    'idq.ms_cycles',
    'idq.ms_dsb_cycles',
    'idq.ms_dsb_occur',
    'idq.ms_dsb_uops',
    'idq.ms_mite_uops',
    'idq.ms_switches',
    'idq.ms_uops',
    'idq_uops_not_delivered.core',
    'idq_uops_not_delivered.cycles_0_uops_deliv.core',
    'idq_uops_not_delivered.cycles_fe_was_ok',
    'idq_uops_not_delivered.cycles_le_1_uop_deliv.core',
    'idq_uops_not_delivered.cycles_le_2_uop_deliv.core',
    'idq_uops_not_delivered.cycles_le_3_uop_deliv.core',
    'hle_retired.aborted',
    'hle_retired.aborted_misc1',
    'hle_retired.aborted_misc2',
    'hle_retired.aborted_misc3',
    'hle_retired.aborted_misc4',
    'hle_retired.aborted_misc5',
    'hle_retired.commit',
    'hle_retired.start',
    'machine_clears.memory_ordering',
    'mem_trans_retired.load_latency_gt_128',
    'mem_trans_retired.load_latency_gt_16',
    'mem_trans_retired.load_latency_gt_256',
    'mem_trans_retired.load_latency_gt_32',
    'mem_trans_retired.load_latency_gt_4',
    'mem_trans_retired.load_latency_gt_512',
    'mem_trans_retired.load_latency_gt_64',
    'mem_trans_retired.load_latency_gt_8',
    'misalign_mem_ref.loads',
    'misalign_mem_ref.stores',
    'offcore_response.all_code_rd.l3_miss.any_response',
    'offcore_response.all_code_rd.l3_miss.local_dram',
    'offcore_response.all_data_rd.l3_miss.any_response',
    'offcore_response.all_data_rd.l3_miss.local_dram',
    'offcore_response.all_reads.l3_miss.any_response',
    'offcore_response.all_reads.l3_miss.local_dram',
    'offcore_response.all_requests.l3_miss.any_response',
    'offcore_response.all_rfo.l3_miss.any_response',
    'offcore_response.all_rfo.l3_miss.local_dram',
    'offcore_response.demand_code_rd.l3_miss.any_response',
    'offcore_response.demand_code_rd.l3_miss.local_dram',
    'offcore_response.demand_data_rd.l3_miss.any_response',
    'offcore_response.demand_data_rd.l3_miss.local_dram',
    'offcore_response.demand_rfo.l3_miss.any_response',
    'offcore_response.demand_rfo.l3_miss.local_dram',
    'offcore_response.pf_l2_code_rd.l3_miss.any_response',
    'offcore_response.pf_l2_data_rd.l3_miss.any_response',
    'offcore_response.pf_l2_rfo.l3_miss.any_response',
    'offcore_response.pf_l3_code_rd.l3_miss.any_response',
    'offcore_response.pf_l3_data_rd.l3_miss.any_response',
    'offcore_response.pf_l3_rfo.l3_miss.any_response',
    'rtm_retired.aborted',
    'rtm_retired.aborted_misc1',
    'rtm_retired.aborted_misc2',
    'rtm_retired.aborted_misc3',
    'rtm_retired.aborted_misc4',
    'rtm_retired.aborted_misc5',
    'rtm_retired.commit',
    'rtm_retired.start',
    'tx_exec.misc1',
    'tx_exec.misc2',
    'tx_exec.misc3',
    'tx_exec.misc4',
    'tx_exec.misc5',
    'tx_mem.abort_capacity_write',
    'tx_mem.abort_conflict',
    'tx_mem.abort_hle_elision_buffer_mismatch',
    'tx_mem.abort_hle_elision_buffer_not_empty',
    'tx_mem.abort_hle_elision_buffer_unsupported_alignment',
    'tx_mem.abort_hle_store_to_elided_lock',
    'tx_mem.hle_elision_buffer_full',
    'cpl_cycles.ring0',
    'cpl_cycles.ring0_trans',
    'cpl_cycles.ring123',
    'lock_cycles.split_lock_uc_lock_duration',
    'arith.divider_uops',
    'baclears.any',
    'br_inst_exec.all_branches',
    'br_inst_exec.all_conditional',
    'br_inst_exec.all_direct_jmp',
    'br_inst_exec.all_direct_near_call',
    'br_inst_exec.all_indirect_jump_non_call_ret',
    'br_inst_exec.all_indirect_near_return',
    'br_inst_exec.nontaken_conditional',
    'br_inst_exec.taken_conditional',
    'br_inst_exec.taken_direct_jump',
    'br_inst_exec.taken_direct_near_call',
    'br_inst_exec.taken_indirect_jump_non_call_ret',
    'br_inst_exec.taken_indirect_near_call',
    'br_inst_exec.taken_indirect_near_return',
    'br_inst_retired.all_branches',
    'br_inst_retired.all_branches_pebs',
    'br_inst_retired.conditional',
    'br_inst_retired.far_branch',
    'br_inst_retired.near_call',
    'br_inst_retired.near_call_r3',
    'br_inst_retired.near_return',
    'br_inst_retired.near_taken',
    'br_inst_retired.not_taken',
    'br_misp_exec.all_branches',
    'br_misp_exec.all_conditional',
    'br_misp_exec.all_indirect_jump_non_call_ret',
    'br_misp_exec.nontaken_conditional',
    'br_misp_exec.taken_conditional',
    'br_misp_exec.taken_indirect_jump_non_call_ret',
    'br_misp_exec.taken_indirect_near_call',
    'br_misp_exec.taken_return_near',
    'br_misp_retired.all_branches',
    'br_misp_retired.all_branches_pebs',
    'br_misp_retired.conditional',
    'br_misp_retired.near_taken',
    'cpu_clk_thread_unhalted.one_thread_active',
    'cpu_clk_thread_unhalted.ref_xclk',
    'cpu_clk_thread_unhalted.ref_xclk_any',
    'cpu_clk_unhalted.one_thread_active',
    'cpu_clk_unhalted.ref_tsc',
    'cpu_clk_unhalted.ref_xclk',
    'cpu_clk_unhalted.ref_xclk_any',
    'cpu_clk_unhalted.thread',
    'cpu_clk_unhalted.thread_any',
    'cpu_clk_unhalted.thread_p',
    'cpu_clk_unhalted.thread_p_any',
    'cycle_activity.cycles_l1d_pending',
    'cycle_activity.cycles_l2_pending',
    'cycle_activity.cycles_ldm_pending',
    'cycle_activity.cycles_no_execute',
    'cycle_activity.stalls_l1d_pending',
    'cycle_activity.stalls_l2_pending',
    'cycle_activity.stalls_ldm_pending',
    'ild_stall.iq_full',
    'ild_stall.lcp',
    'inst_retired.any',
    'inst_retired.any_p',
    'inst_retired.prec_dist',
    'inst_retired.x87',
    'int_misc.recovery_cycles',
    'int_misc.recovery_cycles_any',
    'ld_blocks.no_sr',
    'ld_blocks.store_forward',
    'ld_blocks_partial.address_alias',
    'load_hit_pre.hw_pf',
    'load_hit_pre.sw_pf',
    'lsd.cycles_4_uops',
    'lsd.cycles_active',
    'lsd.uops',
    'machine_clears.count',
    'machine_clears.cycles',
    'machine_clears.maskmov',
    'machine_clears.smc',
    'move_elimination.int_eliminated',
    'move_elimination.int_not_eliminated',
    'move_elimination.simd_eliminated',
    'move_elimination.simd_not_eliminated',
    'other_assists.any_wb_assist',
    'resource_stalls.any',
    'resource_stalls.rob',
    'resource_stalls.rs',
    'resource_stalls.sb',
    'rob_misc_events.lbr_inserts',
    'rs_events.empty_cycles',
    'rs_events.empty_end',
    'uops_dispatched_port.port_0',
    'uops_dispatched_port.port_1',
    'uops_dispatched_port.port_2',
    'uops_dispatched_port.port_3',
    'uops_dispatched_port.port_4',
    'uops_dispatched_port.port_5',
    'uops_dispatched_port.port_6',
    'uops_dispatched_port.port_7',
    'uops_executed.core',
    'uops_executed.core_cycles_ge_1',
    'uops_executed.core_cycles_ge_2',
    'uops_executed.core_cycles_ge_3',
    'uops_executed.core_cycles_ge_4',
    'uops_executed.core_cycles_none',
    'uops_executed.cycles_ge_1_uop_exec',
    'uops_executed.cycles_ge_2_uops_exec',
    'uops_executed.cycles_ge_3_uops_exec',
    'uops_executed.cycles_ge_4_uops_exec',
    'uops_executed.stall_cycles',
    'uops_executed_port.port_0',
    'uops_executed_port.port_0_core',
    'uops_executed_port.port_1',
    'uops_executed_port.port_1_core',
    'uops_executed_port.port_2',
    'uops_executed_port.port_2_core',
    'uops_executed_port.port_3',
    'uops_executed_port.port_3_core',
    'uops_executed_port.port_4',
    'uops_executed_port.port_4_core',
    'uops_executed_port.port_5',
    'uops_executed_port.port_5_core',
    'uops_executed_port.port_6',
    'uops_executed_port.port_6_core',
    'uops_executed_port.port_7',
    'uops_executed_port.port_7_core',
    'uops_issued.any',
    'uops_issued.core_stall_cycles',
    'uops_issued.flags_merge',
    'uops_issued.single_mul',
    'uops_issued.slow_lea',
    'uops_issued.stall_cycles',
    'uops_retired.all',
    'uops_retired.core_stall_cycles',
    'uops_retired.retire_slots',
    'uops_retired.stall_cycles',
    'uops_retired.total_cycles',
    'dtlb_load_misses.miss_causes_a_walk',
    'dtlb_load_misses.pde_cache_miss',
    'dtlb_load_misses.stlb_hit',
    'dtlb_load_misses.stlb_hit_2m',
    'dtlb_load_misses.stlb_hit_4k',
    'dtlb_load_misses.walk_completed',
    'dtlb_load_misses.walk_completed_1g',
    'dtlb_load_misses.walk_completed_2m_4m',
    'dtlb_load_misses.walk_completed_4k',
    'dtlb_load_misses.walk_duration',
    'dtlb_store_misses.miss_causes_a_walk',
    'dtlb_store_misses.pde_cache_miss',
    'dtlb_store_misses.stlb_hit',
    'dtlb_store_misses.stlb_hit_2m',
    'dtlb_store_misses.stlb_hit_4k',
    'dtlb_store_misses.walk_completed',
    'dtlb_store_misses.walk_completed_1g',
    'dtlb_store_misses.walk_completed_2m_4m',
    'dtlb_store_misses.walk_completed_4k',
    'dtlb_store_misses.walk_duration',
    'ept.walk_cycles',
    'itlb.itlb_flush',
    'itlb_misses.miss_causes_a_walk',
    'itlb_misses.stlb_hit',
    'itlb_misses.stlb_hit_2m',
    'itlb_misses.stlb_hit_4k',
    'itlb_misses.walk_completed',
    'itlb_misses.walk_completed_1g',
    'itlb_misses.walk_completed_2m_4m',
    'itlb_misses.walk_completed_4k',
    'itlb_misses.walk_duration',
    'page_walker_loads.dtlb_l1',
    'page_walker_loads.dtlb_l2',
    'page_walker_loads.dtlb_l3',
    'page_walker_loads.dtlb_memory',
    'page_walker_loads.ept_dtlb_l1',
    'page_walker_loads.ept_dtlb_l2',
    'page_walker_loads.ept_dtlb_l3',
    'page_walker_loads.ept_dtlb_memory',
    'page_walker_loads.ept_itlb_l1',
    'page_walker_loads.ept_itlb_l2',
    'page_walker_loads.ept_itlb_l3',
    'page_walker_loads.ept_itlb_memory',
    'page_walker_loads.itlb_l1',
    'page_walker_loads.itlb_l2',
    'page_walker_loads.itlb_l3',
    'page_walker_loads.itlb_memory',
    'tlb_flush.dtlb_thread',
    'tlb_flush.stlb_any'
]

def sample_to_line(sample, use_pid):
    if not use_pid:
        line = sample['program']
    else:
        line = f"{sample['program']} [{sample['pid']}]"
    for i in range(len(sample['sources'])-1, -1, -1):
        line = f"{line};{sample['sources'][i]['name']}"
    return line

def samples_to_stackcollapse(samples, const = 0.0, use_pid = False):
    lines = dict()

    for s in samples:
        key = sample_to_line(s, use_pid)
        if not key in lines.keys():
            lines[key] = s['power'] * s['T'] + 0.0
        else:
            lines[key] += s['power'] * s['T'] + 0.0

    return lines

def print_lines(lines, positive):
    s_lines = sorted(lines.keys())
    for k in s_lines:
        if not positive or lines[k] > 0:
            print(f"{k} {lines[k]:20f}")

def select_features(base_features, setA, setB):
    new_f = []
    for f in base_features:
        if f in setA and f in setB:
            new_f.append(f)
    return new_f

def split_dataset_Xy(df, y_name, features):
    y = df[y_name]
    X = df.div(df['T'], axis = 0)
    X = X.reindex(features, axis=1, fill_value=0)
    return X,y

def train_model(df, features, y, positive = False):
    X,y = split_dataset_Xy(df, y, features)
    model = LinearRegression(positive = positive)
    model = model.fit(X,y)
    model.intercept_ = 0
    return model

def run(args):
    conv = ScriptConverter(args.filename, cpu=not args.no_cpu)
    df = conv.to_dataframe()

    training_df = pd.read_csv(args.training, sep=args.sep)
    new_features = select_features(features, conv.features(), training_df.columns)
    model = train_model(training_df, new_features, 'power/energy-pkg/', positive = args.positive)
    training_df = None

    df = df.drop(['program', 'pid'], axis=1)
    df = df.div(df['T'], axis = 0)
    X = df.reindex(new_features, axis=1, fill_value=0)

    y_pred = model.predict(X)

    samples = conv.timer.samples
    for i in range(len(samples)):
        samples[i]['power'] = y_pred[i]

    y_pred = None

    lines = samples_to_stackcollapse(samples, 0.0, args.pid)
    print_lines(lines, args.positive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert script to csv")
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file name')
    parser.add_argument('--sep', type=str, default=';', help='Field separator')
    parser.add_argument('-t', '--training', type=str)
    parser.add_argument('-y', type=str, default='power/energy-pkg/')
    parser.add_argument('--no-cpu', action='store_true')
    parser.add_argument('--add-const', type=float, default=0.0)
    parser.add_argument('--positive', action='store_true')
    parser.add_argument('--pid', action='store_true')
    args = parser.parse_args()

    run(args)

    # parser = ScriptParser("./data/examples/test.txt")
    # parser.parse()
    # lines = samples_to_stackcollapse(parser.samples)
    # print_lines(lines)
