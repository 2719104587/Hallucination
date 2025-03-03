# af2 network module
import mock

# AF2-specific libraries
from hallucination.model.alphafold2.protein import from_prediction
from hallucination.model.alphafold2.msa import make_sequence_features, make_msa_features
from hallucination.model.alphafold2.template import mk_mock_template
from hallucination.model.alphafold2 import model


def af2_predict(
    query_sequence,
    idx,
    model_runner: model.RunModel,
    random_seed=0,
    external_template=None,
):
    """Predicts structure for a given oligomer using AlphaFold2."""
    # Mock pipeline.
    data_pipeline_mock = mock.Mock()

    # Get features.
    if external_template is None:
        data_pipeline_mock.process.return_value = {
            **make_sequence_features(
                sequence=query_sequence, description="none", num_res=len(query_sequence)
            ),
            **make_msa_features(
                msas=[[query_sequence]], deletion_matrices=[[[0] * len(query_sequence)]]
            ),
            **mk_mock_template(query_sequence),
        }
        feature_dict = data_pipeline_mock.process()
    else:
        data_pipeline_mock.process.return_value = {
            **make_sequence_features(
                sequence=query_sequence, description="none", num_res=len(query_sequence)
            ),
            **make_msa_features(
                msas=[[query_sequence]], deletion_matrices=[[[0] * len(query_sequence)]]
            ),
        }
        feature_dict = data_pipeline_mock.process()

        for template_name, template_value in external_template.items():
            feature_dict[template_name] = template_value

    feature_dict["residue_index"] = idx.float().numpy()

    # Run AlphaFold2 prediction.
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=random_seed
    )
    prediction_results = model_runner.predict(processed_feature_dict)
    unrelaxed_protein = from_prediction(processed_feature_dict, prediction_results)
    # scale pLDDT to be between 0 and 1
    prediction_results["plddt"] = prediction_results["plddt"] / 100.0

    return prediction_results, unrelaxed_protein
