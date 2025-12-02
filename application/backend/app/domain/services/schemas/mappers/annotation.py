from domain.db.models import AnnotationDB
from domain.services.schemas.annotation import AnnotationSchema


def annotations_db_to_schemas(annotations_db: list[AnnotationDB]) -> list[AnnotationSchema]:
    """
    Convert a list of AnnotationDB instances to AnnotationSchema objects.
    """
    return [
        AnnotationSchema.model_validate({"config": annotation_db.config, "label_id": annotation_db.label_id})
        for annotation_db in annotations_db
    ]
