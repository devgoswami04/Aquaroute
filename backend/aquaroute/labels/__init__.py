"""Module 3 — ground-truth flood labelling from SAR (or synthetic) + reports."""
from aquaroute.labels.merge import merge_report_labels
from aquaroute.labels.sar_labels import label_event_from_sar
from aquaroute.labels.training_set import assemble_training_set

__all__ = ["label_event_from_sar", "merge_report_labels", "assemble_training_set"]
