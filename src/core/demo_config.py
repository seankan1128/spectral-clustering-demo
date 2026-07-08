"""Demo-specific configuration and reporting helpers."""

PROJECT_NAME = 'Spectral Clustering Demo'
VERSION = '1.0.0'

SUPPORTED_DATASETS = ('mnist', 'reuters', 'cc', 'cc_semisup')


def print_banner(dataset):
    print('=' * 60)
    print(f'{PROJECT_NAME} v{VERSION}')
    print(f'Dataset: {dataset}')
    print('=' * 60)


def format_evaluation_report(accuracy, nmi_score, n_clusters, extra_label=''):
    label = f' ({extra_label})' if extra_label else ''
    return (
        f'\n--- Evaluation{label} ---\n'
        f'  Clusters: {n_clusters}\n'
        f'  Accuracy: {accuracy:.3f}\n'
        f'  NMI:      {nmi_score:.3f}\n'
        f'------------------------'
    )
