"""Run from your own prepared data; fictional relative paths only."""
from tcia_prostate_rigid import register_pair, RegistrationConfig

if __name__ == '__main__':
    result = register_pair(
        'data/mr/image.nrrd', 'data/us/image.nrrd',
        'data/mr/prostate_label.nrrd', 'data/us/prostate_label.nrrd',
        'outputs/case-001',
        mr_targets={1: 'data/mr/target_1_label.nrrd'},
        us_targets={1: 'data/us/target_1_label.nrrd'},
        config=RegistrationConfig(),
    )
    print(result.metrics)

