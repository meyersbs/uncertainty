from pkg_resources import resource_filename

BCLASS_CLASSIFIER_PATH = resource_filename('uncertainty', 'models/bclass.p')
MCLASS_CLASSIFIER_PATH = resource_filename('uncertainty', 'models/mclass.p')
VECTORIZER_PATH = resource_filename('uncertainty', 'vectorizers/vectorizer.p')

UNCERTAINTY_CLASS_MAP = {
        'speculation_modal_probable_': 'E',
        'speculation_hypo_doxastic _': 'D',
        'speculation_hypo_condition _': 'N',
        'speculation_hypo_investigation _': 'I',
        'O': 'C'
    }
