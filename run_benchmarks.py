import argparse
import os
from config import ProjectConfig
from sparse_attention.runtime_backend import require_tpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sparse Attention Benchmarks")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--hbm", action="store_true")
    parser.add_argument("--flops", action="store_true")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--roofline", action="store_true")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--viz-only", action="store_true")
    parser.add_argument("--mega-stress", action="store_true")
    parser.add_argument("--target-gb", type=float, default=40.0)
    parser.add_argument("--viz-all", action="store_true")
    parser.add_argument("--no-pallas", action="store_true")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--sparsity",
        type=str,
        default="combined",
        choices=["causal", "strided", "fixed", "random", "combined"],
    )
    parser.add_argument("--output-dir", type=str, default="results")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = ProjectConfig.for_quick_test() if args.quick else ProjectConfig()
    use_pallas = not args.no_pallas

    if args.hbm:
        from benchmarks.profile_hbm import profile_hbm_sweep, print_hbm_report

        print("HBM profiling is running.")
        try:
            require_tpu("HBM profiling")
        except RuntimeError as exc:
            print(str(exc))
            return
        profiles = profile_hbm_sweep(
            seq_lengths=cfg.benchmark.quick_seq_lengths if args.quick else cfg.benchmark.seq_lengths,
            batch_sizes=cfg.benchmark.quick_batch_sizes if args.quick else cfg.benchmark.batch_sizes,
            block_size=args.block_size,
            sparsity_type=args.sparsity,
        )
        print_hbm_report(profiles)
        return

    if args.flops:
        from benchmarks.profile_flops import profile_flops_detailed, print_flops_report

        print("FLOPs profiling is running.")
        try:
            require_tpu("FLOPs profiling")
        except RuntimeError as exc:
            print(str(exc))
            return
        seq_lens = cfg.benchmark.quick_seq_lengths if args.quick else cfg.benchmark.seq_lengths
        profiles = [profile_flops_detailed(4, seq_len, args.block_size, args.sparsity) for seq_len in seq_lens]
        print_flops_report(profiles)
        return

    if args.scaling:
        from benchmarks.scaling_analysis import scaling_analysis, print_scaling_report

        print("Scaling analysis is running.")
        try:
            require_tpu("Scaling analysis")
        except RuntimeError as exc:
            print(str(exc))
            return
        results = scaling_analysis(
            [256, 512, 1024, 2048] if args.quick else [256, 512, 1024, 2048, 4096],
            args.block_size,
            args.sparsity,
        )
        print_scaling_report(results)
        return

    if args.roofline:
        from benchmarks.roofline import roofline_analysis

        print("Roofline analysis is running.")
        try:
            roofline_analysis(
                seq_lengths=[256, 512, 1024, 2048] if args.quick else [256, 512, 1024, 2048, 4096],
                use_pallas=use_pallas,
                output_dir=args.output_dir,
            )
        except RuntimeError as exc:
            print(str(exc))
        return

    if args.stress:
        from benchmarks.stress_test import run_all_stress_tests

        print("Stress suite is running.")
        try:
            require_tpu("Stress suite")
        except RuntimeError as exc:
            print(str(exc))
            return
        run_all_stress_tests(output_dir=args.output_dir, use_pallas=use_pallas, quick=args.quick)
        return

    if args.viz_only:
        from sparse_attention.plotly_dashboard import generate_plotly_dashboard
        from sparse_attention.visualize import generate_all_plots

        print("Exporting saved visuals.")
        json_path = os.path.join(args.output_dir, "benchmark_results.json")
        stress_path = os.path.join(args.output_dir, "stress_test_all.json")
        generate_all_plots(json_path, os.path.join(args.output_dir, "plots"))
        generate_plotly_dashboard(json_path, stress_path, os.path.join(args.output_dir, "plotly_plots"))
        return

    if args.mega_stress:
        from benchmarks.mega_stress import run_mega_stress
        from sparse_attention.viz_stress import generate_stress_viz

        print(f"Mega stress is running with host target {args.target_gb:.1f}GB.")
        try:
            require_tpu("Mega stress")
        except RuntimeError as exc:
            print(str(exc))
            return
        run_mega_stress(args.output_dir, use_pallas, args.target_gb)
        mega_path = os.path.join(args.output_dir, "mega_stress_all.json")
        stress_path = os.path.join(args.output_dir, "stress_test_all.json")
        viz_dir = os.path.join(args.output_dir, "viz_stress")
        try:
            generate_stress_viz(stress_path, mega_path, viz_dir)
        except Exception as exc:
            print(f"Visualization export skipped: {exc}")
        return

    if args.viz_all:
        from sparse_attention.viz_memory import generate_memory_viz
        from sparse_attention.viz_scaling import generate_scaling_viz
        from sparse_attention.viz_stress import generate_stress_viz
        from sparse_attention.viz_throughput import generate_throughput_viz
        from sparse_attention.viz_training import generate_training_viz

        print("Exporting visualization bundles.")
        benchmark_path = os.path.join(args.output_dir, "benchmark_results.json")
        stress_path = os.path.join(args.output_dir, "stress_test_all.json")
        mega_path = os.path.join(args.output_dir, "mega_stress_all.json")
        roofline_path = os.path.join(args.output_dir, "roofline_analysis.json")

        try:
            generate_throughput_viz(benchmark_path, os.path.join(args.output_dir, "viz_throughput"))
        except Exception:
            pass
        try:
            generate_memory_viz(benchmark_path, os.path.join(args.output_dir, "viz_memory"))
        except Exception:
            pass
        try:
            generate_stress_viz(stress_path, mega_path, os.path.join(args.output_dir, "viz_stress"))
        except Exception:
            pass
        try:
            generate_scaling_viz(roofline_path, os.path.join(args.output_dir, "viz_scaling"))
        except Exception:
            pass
        try:
            generate_training_viz("tensorboard_logs", os.path.join(args.output_dir, "viz_training"))
        except Exception:
            pass
        print("Visualization export finished.")
        return

    from benchmarks.benchmark_suite import run_benchmark_suite, run_quick_benchmark
    from sparse_attention.plotly_dashboard import generate_plotly_dashboard
    from sparse_attention.visualize import generate_all_plots

    print(
        f"Benchmark suite is running with sparsity={args.sparsity}, "
        f"pallas={'on' if use_pallas else 'off'}, block_size={args.block_size}."
    )
    try:
        require_tpu("Benchmark suite")
    except RuntimeError as exc:
        print(str(exc))
        return
    if args.quick:
        comparisons, _ = run_quick_benchmark(args.block_size, args.sparsity, use_pallas)
    else:
        comparisons, _ = run_benchmark_suite(
            cfg.benchmark.seq_lengths,
            cfg.benchmark.batch_sizes,
            args.block_size,
            args.sparsity,
            cfg.benchmark.n_warmup,
            cfg.benchmark.n_iterations,
            use_pallas,
            args.output_dir,
        )

    if comparisons:
        avg = sum(comparison.hbm_reduction_pct for comparison in comparisons) / len(comparisons)
        print(f"Average HBM reduction: {avg:.1f}%")
        json_path = os.path.join(args.output_dir, "benchmark_results.json")
        stress_path = os.path.join(args.output_dir, "stress_test_all.json")
        try:
            generate_all_plots(json_path, os.path.join(args.output_dir, "plots"))
            generate_plotly_dashboard(json_path, stress_path, os.path.join(args.output_dir, "plotly_plots"))
        except Exception as exc:
            print(f"Visualization export skipped: {exc}")


if __name__ == "__main__":
    main()
