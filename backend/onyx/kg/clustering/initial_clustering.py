from typing import cast

from rapidfuzz.fuzz import ratio
from sqlalchemy import text

from onyx.configs.kg_configs import KG_CLUSTERING_RETRIVE_THRESHOLD
from onyx.configs.kg_configs import KG_CLUSTERING_THRESHOLD
from onyx.db.document import update_document_kg_info
from onyx.db.engine import get_session_with_current_tenant
from onyx.db.entities import add_or_update_entity
from onyx.db.entities import delete_entities_by_id_names
from onyx.db.entities import Document
from onyx.db.entities import get_entities_by_grounding
from onyx.db.entities import KGEntity
from onyx.db.entities import KGEntityExtractionStaging
from onyx.db.relationships import add_relationship
from onyx.db.relationships import add_relationship_type
from onyx.db.relationships import delete_relationship_types_by_id_names
from onyx.db.relationships import delete_relationships_by_id_names
from onyx.db.relationships import get_all_relationship_types
from onyx.db.relationships import get_all_relationships
from onyx.kg.models import KGGroundingType
from onyx.kg.models import KGStage
from onyx.utils.logger import setup_logger

# from sklearn.cluster import SpectralClustering  # type: ignore

logger = setup_logger()


def _transfer_one_grounded_entity(
    entity: KGEntityExtractionStaging,
) -> str | None:
    """
    Transfer a single grounded KGEntityExtractionStaging to KGEntity.
    The entity will be merged with another KGEntity if it is sufficiently similar.
    The representative of the cluster will be the one with the document_id, unless
    none of the staging entities in the cluster have a document_id.

    Returns the id_name of the added/updated entity if successful, otherwise None.
    """
    # find candidates to match against
    with get_session_with_current_tenant() as db_session:
        # get clustering_name and filtering conditions
        if entity.document_id is None:
            entity_clustering_name = entity.name.lower()
            filters = []  # can match with any KGEntity
        else:
            entity_clustering_name: str = (
                db_session.query(Document.semantic_id)
                .filter(Document.id == entity.document_id)
                .scalar()
            ).lower()
            # can only match with entities without a document_id
            filters = [KGEntity.document_id.is_(None)]

        # find entities with a similar name to merge with
        similar_entities = []
        if not any(char.isdigit() for char in entity_clustering_name):
            db_session.execute(
                text(
                    f"SET pg_trgm.similarity_threshold = {KG_CLUSTERING_RETRIVE_THRESHOLD}"
                )
            )
            similar_entities = (
                db_session.query(KGEntity)
                .filter(
                    # find entities of the same type with a similar name
                    *filters,
                    KGEntity.entity_type_id_name == entity.entity_type_id_name,
                    KGEntity.clustering_name.op("%")(entity_clustering_name),
                )
                .all()
            )

    # assign them to the nearest cluster if we're confident they're the same entity
    best_score = -1.0
    best_entity = None
    for similar in similar_entities:
        # skip those with numbers so we don't cluster version1 and version2, etc.
        if any(char.isdigit() for char in similar.clustering_name):
            continue
        score = ratio(similar.clustering_name, entity_clustering_name)
        if score >= KG_CLUSTERING_THRESHOLD * 100 and score > best_score:
            best_score = score
            best_entity = similar

    with get_session_with_current_tenant() as db_session:
        entity_name = entity.name
        entity_type = entity.entity_type_id_name
        entity_document_id = entity.document_id
        entity_occurrences = entity.occurrences or 1
        entity_attributes = entity.attributes or {}
        entity_alternative_names = set(entity.alternative_names or [])

        # if we found a match, update the existing entity
        if best_entity:
            logger.info(f"Merging {entity.id_name} with {best_entity.id_name}")
            entity_name = best_entity.name
            entity_document_id = best_entity.document_id or entity_document_id
            entity_occurrences += best_entity.occurrences or 1
            entity_attributes.update(best_entity.attributes or {})
            entity_alternative_names.update(best_entity.alternative_names or [])
            entity_alternative_names.add(entity.name)

        # create/update the entity
        transferred_entity = add_or_update_entity(
            db_session=db_session,
            kg_stage=KGStage.NORMALIZED,
            entity_type=entity_type,
            name=entity_name,
            document_id=entity_document_id,
            occurrences=entity_occurrences,
            attributes=entity_attributes,
            alternative_names=list(entity_alternative_names),
        )
        db_session.commit()

    return transferred_entity.id_name if transferred_entity else None


def kg_clustering(
    tenant_id: str, index_name: str, processing_chunk_batch_size: int = 8
) -> None:
    """
    Here we will cluster the extractions based on their cluster frameworks.
    Initially, this will only focus on grounded entities with pre-determined
    relationships, so 'clustering' is actually not yet required.
    However, we may need to reconcile entities coming from different sources.

    The primary purpose of this function is to populate the actual KG tables
    from the temp_extraction tables.

    This will change with deep extraction, where grounded-sourceless entities
    can be extracted and then need to be clustered.
    """

    logger.info(f"Starting kg clustering for tenant {tenant_id}")

    ## Retrieval

    source_documents_w_successful_transfers: set[str] = set()
    source_documents_w_failed_transfers: set[str] = set()

    with get_session_with_current_tenant() as db_session:

        relationship_types = get_all_relationship_types(
            db_session, kg_stage=KGStage.EXTRACTED
        )

        relationships = get_all_relationships(db_session, kg_stage=KGStage.EXTRACTED)

        grounded_entities: list[KGEntityExtractionStaging] = cast(
            list[KGEntityExtractionStaging],
            get_entities_by_grounding(
                db_session, KGStage.EXTRACTED, KGGroundingType.GROUNDED
            ),
        )

    ## Clustering

    # TODO: re-implement clustering of ungrounded entities as well as
    # grounded entities that do not have a source document with deep extraction enabled!
    # For now we would just dedupe grounded entities that have very similar names
    # This will be reimplemented when deep extraction is enabled.

    transferred_entities: list[str] = []
    cluster_translations: dict[str, str] = {}

    # transfer the initial grounded entities
    for entity in grounded_entities:
        transferred_entity_id_name = _transfer_one_grounded_entity(entity)
        if transferred_entity_id_name:
            transferred_entities.append(entity.id_name)
            cluster_translations[entity.id_name] = transferred_entity_id_name

    ## Database operations

    transferred_relationship_types: list[str] = []
    for relationship_type in relationship_types:
        with get_session_with_current_tenant() as db_session:
            added_relationship_type_id_name = add_relationship_type(
                db_session,
                KGStage.NORMALIZED,
                source_entity_type=relationship_type.source_entity_type_id_name,
                relationship_type=relationship_type.type,
                target_entity_type=relationship_type.target_entity_type_id_name,
                extraction_count=relationship_type.occurrences or 1,
            )
            db_session.commit()
            transferred_relationship_types.append(added_relationship_type_id_name)

    transferred_relationships: list[str] = []
    for relationship in relationships:
        with get_session_with_current_tenant() as db_session:
            try:
                # update the id_name
                (
                    source_entity_id_name,
                    relationship_string,
                    target_entity_id_name,
                ) = relationship.id_name.split("__")

                new_relationship_id_name = "__".join(
                    (
                        cluster_translations.get(
                            source_entity_id_name, source_entity_id_name
                        ),
                        relationship_string,
                        cluster_translations.get(
                            target_entity_id_name, target_entity_id_name
                        ),
                    )
                )
                add_relationship(
                    db_session,
                    KGStage.NORMALIZED,
                    relationship_id_name=new_relationship_id_name,
                    source_document_id=relationship.source_document or "",
                    occurrences=relationship.occurrences or 1,
                )

                if relationship.source_document:
                    source_documents_w_successful_transfers.add(
                        relationship.source_document
                    )
                db_session.commit()
                transferred_relationships.append(relationship.id_name)

            except Exception as e:
                if relationship.source_document:
                    source_documents_w_failed_transfers.add(
                        relationship.source_document
                    )
                logger.error(
                    f"Error transferring relationship {relationship.id_name}: {e}"
                )

    # TODO: remove the /relationship types & entities that correspond to relationships
    # source documents that failed to transfer. I.e, do a proper rollback

    # TODO: update Vespa info when clustering/changes are performed

    # delete the added objects from the staging tables

    logger.info(f"Transfered {len(transferred_entities)} entities")
    logger.info(f"Transfered {len(transferred_relationships)} relationships")
    logger.info(f"Transfered {len(transferred_relationship_types)} relationship types")
    0 / 0

    try:
        with get_session_with_current_tenant() as db_session:
            delete_relationships_by_id_names(
                db_session, transferred_relationships, kg_stage=KGStage.EXTRACTED
            )
            db_session.commit()
    except Exception as e:
        logger.error(f"Error deleting relationships: {e}")

    try:
        with get_session_with_current_tenant() as db_session:
            delete_relationship_types_by_id_names(
                db_session, transferred_relationship_types, kg_stage=KGStage.EXTRACTED
            )
            db_session.commit()
    except Exception as e:
        logger.error(f"Error deleting relationship types: {e}")

    try:
        with get_session_with_current_tenant() as db_session:
            delete_entities_by_id_names(
                db_session, transferred_entities, kg_stage=KGStage.EXTRACTED
            )
            db_session.commit()
    except Exception as e:
        logger.error(f"Error deleting entities: {e}")

    # Update document kg info

    # with get_session_with_current_tenant() as db_session:
    #     all_kg_extracted_documents_info = get_all_kg_extracted_documents_info(
    #         db_session
    #     )

    for document_id in source_documents_w_successful_transfers:

        # Update the document kg info
        with get_session_with_current_tenant() as db_session:
            update_document_kg_info(
                db_session,
                document_id=document_id,
                kg_stage=KGStage.NORMALIZED,
            )
            db_session.commit()
