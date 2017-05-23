import pkg_resources

DATA_FILE_PATH = 'uncertainty/data/merged_data'
BIN_CUE_MODEL_PATH = pkg_resources.resource_filename(
        'uncertainty', 'models/bclass.p'
    )
BIN_CUE_VECTORIZER_PATH = pkg_resources.resource_filename(
        'uncertainty', 'vectorizers/vectorizer.p'
    )
BIN_SENT_MODEL_PATH = pkg_resources.resource_filename(
        'uncertainty', 'models/binary-sent-model.p'
    )
BIN_SENT_VECTORIZER_PATH = pkg_resources.resource_filename(
        'uncertainty', 'vectorizers/binary-sent-vectorizer.p'
    )
MULTI_CUE_MODEL_PATH = pkg_resources.resource_filename(
        'uncertainty', 'models/mclass.p'
    )
MULTI_CUE_VECTORIZER_PATH = pkg_resources.resource_filename(
        'uncertainty', 'vectorizers/multiclass-cue-vectorizer.p'
    )
MULTI_SENT_MODEL_PATH = pkg_resources.resource_filename(
        'uncertainty', 'models/multiclass-sent-model.p'
    )
MULTI_SENT_VECTORIZER_PATH = pkg_resources.resource_filename(
        'uncertainty', 'vectorizers/multiclass-sent-vectorizer.p'
    )
