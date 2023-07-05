"""Wrapper around Meilisearch vector database."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger(__name__)

class Meilisearch(VectorStore):
    """Wrapper around Meilisearch vector database.

    To use, you should have the ``meilisearch`` python package installed.

    Example:
        .. code-block:: python

            from langchain.vectorstores import Meilisearch
            from langchain.embeddings.openai import OpenAIEmbeddings
            import meilisearch

            # api_key is optional; provide it if your meilisearch instance requires it
            client = meilisearch.Client('http://127.0.0.1:7700', api_key='***')
            index = client.index('langchain_demo')
            embeddings = OpenAIEmbeddings()
            vectorstore = Meilisearch(index, embeddings.embed_query, "text")
    """

    def __init__(
        self,
        index: Any,
        embedding_function: Callable,
        text_key: str,
    ):
        """Initialize with Meilisearch client."""
        try:
            import meilisearch
        except ImportError:
            raise ValueError(
                "Could not import meilisearch python package. "
                "Please install it with `pip install meilisearch`."
            )
        if not isinstance(index, meilisearch.index.Index):
            raise ValueError(
                f"index should be an instance of meilisearch.index.Index, "
                f"got {type(index)}"
            )
        self._index = index
        self._embedding_function = embedding_function
        self._text_key = text_key

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embedding and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadata associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        # from meilisearch.errors import MeiliSearchApiError

        logger.info("Adding %d texts to Meilisearch", len(texts))

        # Embed and create the documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        for i, text in enumerate(texts):
            embedding = self._embedding_function(text)
            metadata = metadatas[i] if metadatas else {}
            metadata[self._text_key] = text
            docs.append({"id": ids[i], "_vector": embedding, "metadata": metadata})

        # send to Meilisearch
        self._index.add_documents(docs)
        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """ Return meilisearch documents most similar to the query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Dictionary of argument(s) to filter on metadata
            index: Index to search on. Default to self._index.

        Returns:
            List of Documents most similar to the query and score for each
        """

        query_obj = self._embedding_function(query)
        docs = []
        results = self._index.search('', {"vector": query_obj, "limit": k})

        for result in results["hits"]:
            metadata = result["metadata"]
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                #score = result["score"]
                #docs.append((Document(page_content=text, metadata=metadata), score))
                docs.append((Document(page_content=text, metadata=metadata), 0.0))
            else:
                logger.warning(
                    f"Found document with no `{self._text_key}` key. Skipping."
                )

        print(docs)
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return meilisearch documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Dictionary of argument(s) to filter on metadata

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_scores = self.similarity_search_with_score(
            query=query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        text_key: str = "text",
        index_name: Optional[str] = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> Meilisearch:
        """Construct Meilisearch wrapper from raw docments

        This is a user friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Meilisearch index.

        This is intended to be a quick way to get started.

        Example:
            Example:
            .. code-block:: python

                from langchain import Meilisearch
                from langchain.embeddings import OpenAIEmbeddings
                import meilisearch

                # The environment should be the one specified next to the API key
                # in your Meilisearch console
                meilisearch.init(api_key="***", environment="...")
                embeddings = OpenAIEmbeddings()
                meilisearch = Meilisearch.from_texts(
                    texts,
                    embeddings,
                    index_name="langchain-demo"
                )
        """
        logger.info("from_texts call")
        try:
            import meilisearch
        except ImportError:
            raise ValueError(
                "Could not import meilisearch python package. "
                "Please install it with `pip install meilisearch`."
            )

        client = meilisearch.Client(meilisearch_url, meilisearch_master_key)

        indexes = client.get_indexes() #check if provided index exists

        if index_name in indexes:
            index = meilisearch.index(index_name)
        elif len(indexes) == 0:
            raise ValueError(
                "No indexes found on Meilisearch."
            )
        else:
            raise ValueError(
                f"Index {index_name} not found on Meilisearch."
                f"Did you mean one of the following indexes: {', '.join(indexes)}"
            )

        for i in range(0, len(texts), batch_size):
            # set end position of batch
            i_end = min(i + batch_size, len(texts))
            # get batch of texts and ids
            lines_batch = texts[i:i_end]
            # create ids if not provided
            if ids:
                ids_batch = ids[i:i_end]
            else:
                ids_batch = [str(uuid.uuid4()) for n in range(i, i_end)]
            # create embeddings
            embeds = embedding.embed_documents(lines_batch)
            # prep metadata and send batch
            if metadatas:
                metadata = metadatas[i:i_end]
            else:
                metadata = [{} for _ in range(i, i_end)]
            for j, line in enumerate(lines_batch):
                metadata[j][text_key] = line
            to_upsert = zip(ids_batch, embeds, metadata)

            # add to Meilisearch
            index.add_documents(documents=list(to_upsert))
        return cls(index, embedding.embed_query, text_key)

    # SHOULD WE KEEP THIS?
    @classmethod
    def from_existing_index(
        cls,
        index_name: str,
        embedding: Embeddings,
        text_key: str = "text",
    ) -> Meilisearch:
        """Load meilisearch vectorstore from index name."""
        try:
            import meilisearch
        except ImportError:
            raise ValueError(
                "Could not import meilisearch python package. "
                "Please install it with `pip install meilisearch-client`."
            )

        return cls(
            meilisearch.Index(index_name, embedding.embed_query, text_key, namespace)
        )
