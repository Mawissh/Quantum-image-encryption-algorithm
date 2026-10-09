import numpy as np
from pathlib import Path

import gneqr_qiskit_4x4 as q4
import quantum_paper_artifacts as qpa
from paper_inputs import paper_inputs
from paper_key_schedule import derive_paper_initial_states
from paper_diffusion import diffusion_keystream_4x4, pbox_from_chaotic_state


def test_self_test_passes():
    checks = q4.run_self_test()
    assert checks["ok"], checks


def test_paper_key_schedule_domain_separates_branches():
    bounds = [(-1.0, 1.0)] * 4
    _, states = derive_paper_initial_states(
        q4.KUMAR_APPENDIX_IMAGE,
        bytes(range(32)),
        bytes(range(12)),
        bounds,
    )
    assert len({states[phase] for phase in ("b", "d", "p")}) == 3


def test_paper_chaotic_inputs_have_expected_shapes_and_bijection():
    params = dict(a=1.0, b=3.5, c=1.0, d=0.6, alpha=1.0, beta=0.02)
    ic = (0.1, 0.1, 0.1, 0.1)
    stream = diffusion_keystream_4x4(
        ic, params, dt=0.01, lag_time=0.1, burn_in_samples=10, decimation=2
    )
    pbox = pbox_from_chaotic_state(
        ic, params, dt=0.01, lag_time=0.1, burn_in_samples=10
    )
    assert stream.shape == (4, 4)
    assert pbox.shape == (4, 4)
    assert np.array_equal(np.sort(pbox.reshape(-1)), np.arange(16))


def test_kumar_pbox_scatter_convention_matches_appendix():
    out = q4.apply_pbox_scatter_image(q4.KUMAR_APPENDIX_PBOX_INPUT_AFTER_XOR)
    assert np.array_equal(out, q4.KUMAR_APPENDIX_PBOX_OUTPUT)


def test_pbox_inverse_restores_input():
    image = q4.KUMAR_APPENDIX_IMAGE
    cipher = q4.apply_pbox_scatter_image(image)
    recovered = q4.apply_inverse_pbox_scatter_image(cipher)
    assert np.array_equal(recovered, image)


def test_paper_verification_status_passes():
    status = q4.verification_status()
    assert status["ok"], status


def test_baker_paper_examples_match():
    status = q4.baker_equation_verification()
    assert status["failure_count"] == 0
    assert status["worked_spatial_m2_n3_k2_x2_y5"] == (3, 2)
    assert status["worked_intensity_c171_k2"] == 234


def test_manuscript_pbox_table_is_permutation_and_inverse():
    table = q4.MANUSCRIPT_PBOX_TABLE
    assert sorted(table.reshape(-1).tolist()) == list(range(16))
    assert np.array_equal(q4.inverse_pbox_table(table), q4.MANUSCRIPT_INVERSE_PBOX_TABLE)

    cipher = q4.apply_pbox_scatter_image(q4.KUMAR_APPENDIX_IMAGE, pbox_table=table)
    recovered = q4.apply_inverse_pbox_scatter_image(cipher, pbox_table=table)
    assert np.array_equal(recovered, q4.KUMAR_APPENDIX_IMAGE)


def test_manuscript_pbox_quantum_matches_classical_pipeline():
    table = q4.MANUSCRIPT_PBOX_TABLE
    expected = q4.classical_pipeline_tables(pbox_table=table)
    cipher_qc = q4.build_encrypt_circuit_4x4(pbox_table=table)
    recovered_qc = q4.build_decrypt_after_encrypt_circuit_4x4(pbox_table=table)

    assert np.array_equal(q4.decode_statevector_image_4x4(cipher_qc), expected["cipher"])
    assert np.array_equal(q4.decode_statevector_image_4x4(recovered_qc), expected["plain"])


def test_stage_circuit_builders_present():
    circuits = q4.build_stage_circuits_4x4(pbox_table=q4.MANUSCRIPT_PBOX_TABLE)
    assert set(circuits) == {
        "gneqr",
        "baker_only",
        "diffusion_only",
        "pbox_only",
        "encrypt",
        "decrypt_after_encrypt",
    }
    assert all(qc.num_qubits == 20 for qc in circuits.values())


def test_resource_tables_include_esop_comparison():
    circuits = q4.build_stage_circuits_4x4(pbox_table=q4.MANUSCRIPT_PBOX_TABLE)
    without_esop = q4.build_stage_circuits_4x4(
        pbox_table=q4.MANUSCRIPT_PBOX_TABLE,
        optimized_esop=False,
    )

    table1 = q4.component_resource_table(circuits)
    table2 = q4.esop_optimization_comparison_table(without_esop, circuits)

    assert "algorithm component" in table1
    assert "CNOT" in table1
    assert "T-depth" in table1
    assert "circuit width" in table1
    assert "CNOT without ESOP" in table2
    assert "CNOT with ESOP" in table2
    assert "Complete encryption" in table2


def test_quantum_paper_artifact_generation(tmp_path):
    manifest = qpa.create_artifacts(tmp_path, pbox_name="manuscript", draw_circuit_png=False)
    assert all(manifest["checks"].values()), manifest["checks"]

    for key in (
        "table_report",
        "fig14",
        "fig15",
        "correlation_scatter",
        "resources",
        "metrics",
        "report",
        "complete_report",
        "complete_report_html",
        "manifest",
    ):
        path = Path(manifest["generated"][key])
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    complete = Path(manifest["generated"]["complete_report"]).read_text(encoding="utf-8")
    assert "Complete Quantum Manuscript Demo Report" in complete
    assert "Fig. 14" in complete
    assert "Table 1: Quantum Resource Metrics By Algorithm Component" in complete
    assert "Table 2: Without vs With ESOP Optimization" in complete
    assert "Circuit Diagrams" in complete

    html = Path(manifest["generated"]["complete_report_html"]).read_text(encoding="utf-8")
    assert "data:image/png;base64" in html
    assert "<table>" in html

    for files in manifest["generated"]["circuits"].values():
        text_path = Path(files["text"])
        assert text_path.exists()
        assert text_path.stat().st_size > 0


def test_paper_inputs_generate_the_same_artifact_set(tmp_path):
    manifest = qpa.create_artifacts(
        tmp_path, inputs=paper_inputs(), draw_circuit_png=False
    )
    assert manifest["pbox_name"] == "generated"
    assert all(manifest["checks"].values())
    assert "input_metadata" in manifest
    assert (tmp_path / "complete_quantum_manuscript_report.html").exists()
    assert (tmp_path / "fig14_quantum_histogram_panels.png").exists()
    assert (tmp_path / "quantum_circuit_resources.md").exists()
