import itertools
import uuid
from abc import abstractmethod
from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)
from tqdm import tqdm

from ..base.module import BaseANN


def create_field_schema(
    column_name: str,
    column_type: str,
    column_has_semantic_index: bool,
    dim: int,
) -> FieldSchema:
    """Create schema for a single field. Assumes type should be a vector of
    floats if `column_has_semantic_index`"""
    if column_has_semantic_index:
        return FieldSchema(
            name=column_name, dtype=DataType.FLOAT_VECTOR, dim=dim
        )
    else:
        dtype = DataType[column_type]
        # Memory in Milvus not pre-allocated so safe to set it to highest max possible
        if dtype == DataType.VARCHAR:
            max_length = 65535
        else:
            max_length = None

        return FieldSchema(
            name=column_name, dtype=dtype, max_length=max_length
        )


def create_schema(
    column_names: List[str],
    column_types: List,
    columns_have_semantic_index: List[bool],
    dim: int,
) -> CollectionSchema:
    """Create schema object, to be used when creating a collection

    Parameters
    ----------
    column_names : List[str]
        The names of the user-provided columns to create. No need to include
        `candidate_task_id`, `sync_id`, or `id`. Should be in the same order
        as `column_types` and `columns_have_semantic_index`.
    column_types : List
        Design for this in milvus TBD. Should be in the same order as
        `column_names` and `columns_have_semantic_index`.
    columns_have_semantic_index : List[bool]
        Denotes whether each field should be indexed. Should be in the same
        order as `column_names` and `column_types`.

    Returns
    -------
    pymilvus.CollectionSchema

    """
    base_fields = [
        # moved to INT64 because of https://github.com/milvus-io/milvus/issues/25843
        # can cause duplicated tasks in resync
        FieldSchema(
            name='id',
            dtype=DataType.INT64,
            is_primary=True,
            # auto_id=True,
        ),
        # FieldSchema(name='sync_id', dtype=DataType.VARCHAR, max_length=50),
        # FieldSchema(
        #     name='candidate_task_id',
        #     dtype=DataType.VARCHAR,
        #     max_length=500,
        # ),
        # FieldSchema(
        #     name='storage_id',
        #     dtype=DataType.VARCHAR,
        #     max_length=500,
        #     is_partition_key=True,
        # ),
    ]
    user_fields = [
        create_field_schema(
            column_name, column_type, column_has_semantic_index, dim
        )
        for column_name, column_type, column_has_semantic_index in zip(
            column_names, column_types, columns_have_semantic_index
        )
    ]
    return CollectionSchema(user_fields + base_fields)


def get_or_create_collection(
    collection_name: str,
    milvus_url: str,
    milvus_username: Optional[str],
    milvus_password: Optional[str],
    column_names: List[str],
    column_types: List[str],
    columns_have_semantic_index: List[bool],
    index_type: str,
    metric: str,
    dim: int,
) -> Collection:
    """Create collection in milvus, and index appropriate columns

    Parameters
    ----------
    collection_name : str
        The name of the collection to create
    milvus_url : str
        The url (including the port) where the milvus server can be accessed,
        e.g. https://in01-1d27f2186307968.gcp-us-west1.vectordb.zillizcloud.com:443
    milvus_username : Optional[str]
        The username to auth to the milvus instance. Can leave out when server
        is running on localhost
    milvus_password : Optional[str]
        The password to auth to the milvus instance. Can leave out then server
        is running on localhost.
    column_names : List[str]
        The names of the user-provided columns to create. No need to include
        `candidate_task_id`, `sync_id`, or `id`. Should be in the same order
        as `column_types` and `columns_have_semantic_index`.
    column_types : List
        Design for this in milvus TBD. Should be in the same order as
        `column_names` and `columns_have_semantic_index`.
    columns_have_semantic_index : List[bool]
        Denotes whether each field should be indexed. Should be in the same
        order as `column_names` and `column_types`.

    """
    _milvus_url = split_milvus_url(milvus_url)

    # connections.connect(
    #     alias='default',
    #     host='localhost',
    #     port='19530',
    #     secure=False,
    #     user='root',
    #     password='Milvus',
    # )
    connections.connect(
        alias='default',
        host='localhost',
        port='19530',
        secure=False,
        user='root',
        password='Milvus',
    )

    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        schema = create_schema(
            column_names,
            column_types,
            columns_have_semantic_index,
            dim,
        )
        collection = Collection(
            name=collection_name,
            schema=schema,
            # num_partitions=MILVUS_DEFAULT_NUM_PARTITIONS,
        )

        # index fields
        DISKANN_INDEX_PARAMS = {
            'index_type': 'DISKANN',
            'metric_type': metric,
        }
        IVF_PQ_INDEX_PARAMS = {
            'index_type': 'IVF_PQ',
            'metric_type': metric,
            'nlist': 4096,
            'm': 10 if dim == 100 else 16,
        }
        IVF_SQ8_INDEX_PARAMS = {
            'index_type': 'IVF_SQ8',
            'metric_type': metric,
            'nlist': 4096,
        }
        INDEX_PARAMS = {
            'DISKANN': DISKANN_INDEX_PARAMS,
            'IVF_PQ': IVF_PQ_INDEX_PARAMS,
            'IVF_SQ8': IVF_SQ8_INDEX_PARAMS,
        }[index_type]

        indexed_fields = [
            column_name
            for column_name, column_has_semantic_index in zip(
                column_names, columns_have_semantic_index
            )
            if column_has_semantic_index
        ]
        for indexed_field in indexed_fields:
            collection.create_index(
                field_name=indexed_field, index_params=INDEX_PARAMS
            )

    # Load collection so it is immediately available to query service
    collection.load()

    return collection


def split_milvus_url(milvus_url):
    """Splits the provided Milvus URL into its components: host, port, and security protocol.

    The function extracts the host name (and optionally port number) from the given URL.
    If no port is specified, the function defaults to 443 for HTTPS and 19530 for HTTP.
    It also identifies whether the connection should be secure based on the scheme.

    Parameters
    ----------
    milvus_url: str
        The Milvus URL to be split. E.g., "https://milvus.co-dev-02.test.com" or "http://10.0.0.3:19530"

    Returns
    -------
    dict
        A dictionary containing:
            - host (str): The host name from the URL.
            - port (int or str): The port number if specified, otherwise the default port based on the scheme.
            - secure (bool): True if the scheme is 'https', otherwise False.
    """
    parsed = urlparse(milvus_url)
    scheme = parsed.scheme
    netloc = parsed.netloc

    if ':' in netloc:
        host, port = netloc.split(':')
    else:
        host = netloc
        if scheme == 'https':
            port = 443
        else:
            port = 19530

    secure = True if scheme == 'https' else False

    return {'host': host, 'port': port, 'secure': secure}


def metric_mapping(_metric: str):
    _metric_type = {'angular': 'cosine', 'euclidean': 'l2'}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f'[Milvus] Not support metric type: {_metric}!!!')
    return _metric_type.upper()


class CustomMilvus(BaseANN):
    """
    Needs `__AVX512F__` flag to run, otherwise the results are incorrect.
    Support HNSW index type
    """

    def __init__(self, metric, dim):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        print(f'>>> Metric:      {self._metric}')
        print(f'>>> Metric Type: {self._metric_type}')
        print(f'>>> Dim:         {self._dim}')

    def fit(self, X):
        def grouper(iterable, n):
            """Collect data into fixed-length chunks or blocks"""
            args = [iter(iterable)] * n
            return itertools.zip_longest(*args, fillvalue=None)

        _X = X.tolist()  # [:1000]
        X_w_idx = list(enumerate(_X))
        _batches = [
            list(filter(lambda item: item is not None, x))
            for x in grouper(X_w_idx, 5000)
        ]

        batches = [
            pd.DataFrame(batch, columns=['id', 'data']) for batch in _batches
        ]
        [self.client.insert(batch[['data', 'id']]) for batch in tqdm(batches)]

    def set_query_arguments(self, ef):
        ...

    @abstractmethod
    def get_search_params(self):
        ...

    def query(self, v, n):
        results = self.client.search(
            data=[v],
            anns_field='data',
            param=self.get_search_params(),
            limit=10,
            offset=0,
            output_fields=['id'],
        )[0]
        return [result.entity.id for result in results]


class CustomIvfPq(CustomMilvus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = get_or_create_collection(
            collection_name=f'IVF_PQ_ANN_Benchmark_{str(uuid.uuid4())[:8]}',
            milvus_url='http://127.0.0.1:19530',
            milvus_username='root',
            milvus_password='Milvus',
            column_names=['data'],
            column_types=['Text'],
            columns_have_semantic_index=[True],
            index_type='IVF_PQ',
            metric=self._metric_type,
            dim=self._dim,
        )

    def get_search_params(self):
        return {
            'params': {
                'nprobe': 16,
            },
            'metric_type': self._metric_type,
        }

    def __str__(self):
        return 'Custom(IVF_PQ)'


class CustomIvfSq8(CustomMilvus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = get_or_create_collection(
            collection_name=f'IVF_SQ8_ANN_Benchmark_{str(uuid.uuid4())[:8]}',
            milvus_url='http://127.0.0.1:19530',
            milvus_username='root',
            milvus_password='Milvus',
            column_names=['data'],
            column_types=['Text'],
            columns_have_semantic_index=[True],
            index_type='IVF_SQ8',
            metric=self._metric_type,
            dim=self._dim,
        )

    def get_search_params(self):
        return {
            'params': {
                'nprobe': 16,
            },
            'metric_type': self._metric_type,
        }

    def __str__(self):
        return 'Custom(IVF_SQ8)'


class CustomDiskann(CustomMilvus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = get_or_create_collection(
            collection_name=f'DISKANN_ANN_Benchmark_{str(uuid.uuid4())[:8]}',
            milvus_url='http://127.0.0.1:19530',
            milvus_username='root',
            milvus_password='Milvus',
            column_names=['data'],
            column_types=['Text'],
            columns_have_semantic_index=[True],
            index_type='DISKANN',
            metric=self._metric_type,
            dim=self._dim,
        )

    def get_search_params(self):
        return {
            'params': {
                'nprobe': 16,
                # "range_filter": min_threshold,
                # "radius": max_threshold,
            },
            'metric_type': self._metric_type,
        }

    def __str__(self):
        return 'Custom(DISKANN)'
