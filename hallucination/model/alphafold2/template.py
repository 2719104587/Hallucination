import numpy as np

from hallucination.model.alphafold2 import residue_constants


def mk_mock_template(query_sequence):
    """Generate mock template features from the input sequence."""

    output_templates_sequence = []
    output_confidence_scores = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    for _ in query_sequence:
        templates_all_atom_positions.append(
            np.zeros((residue_constants.atom_type_num, 3))
        )
        templates_all_atom_masks.append(np.zeros(residue_constants.atom_type_num))
        output_templates_sequence.append("-")
        output_confidence_scores.append(-1)

    output_templates_sequence = "".join(output_templates_sequence)
    templates_aatype = residue_constants.sequence_to_onehot(
        output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID
    )

    template_features = {
        "template_all_atom_positions": np.array(templates_all_atom_positions)[None],
        "template_all_atom_masks": np.array(templates_all_atom_masks)[None],
        "template_sequence": [f"none".encode()],
        "template_aatype": np.array(templates_aatype)[None],
        "template_confidence_scores": np.array(output_confidence_scores)[None],
        "template_domain_names": [f"none".encode()],
        "template_release_date": [f"none".encode()],
    }

    return template_features
