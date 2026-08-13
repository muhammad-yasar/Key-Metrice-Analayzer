# -*- coding: utf-8 -*-
"""
odoo_kma_kpi_model.py
=====================
Add this model to your existing Odoo aspect module.

File location in your Odoo module:
    your_module/models/kma_kpi.py

Also add to your_module/models/__init__.py:
    from . import kma_kpi

And add to your_module/__manifest__.py depends:
    'depends': ['mail', 'base'],  # already there

This creates the kma.kpi model which is:
  - A child of aspect.policies (Many2one)
  - Stores KPI metadata (level, class, metric)
  - Stores a JSON list of Qdrant sentence UUIDs as the bridge
"""

from odoo import fields, models, api
import json
import logging

_logger = logging.getLogger(__name__)


class KmaKpi(models.Model):
    _name        = "kma.kpi"
    _description = "Key Metric / KPI extracted from policy"
    _order       = "policy_id, class_label, level_label"

    # ── Link to existing policy record ────────────────────────────────────
    policy_id = fields.Many2one(
        "aspect.policies",
        string="Policy",
        required=True,
        ondelete="cascade",
        index=True,
    )

    # ── Labels (match your Doccano label names exactly) ────────────────────
    level_label = fields.Selection(
        selection=[
            ("Policy Action",  "Policy Action"),
            ("Policy Outcome", "Policy Outcome"),
            ("Unsure",         "Unsure"),
        ],
        string="Level",
        required=True,
    )

    class_label = fields.Selection(
        selection=[
            ("Area",             "Area"),
            ("Emissions",        "Emissions"),
            ("Site Status",      "Site Status"),
            ("Spending",         "Spending"),
            ("Resource/Report",  "Resource/Report"),
            ("Policy Action",    "Policy Action"),
            ("Miscellaneous",    "Miscellaneous"),
        ],
        string="Class",
        required=True,
    )

    # ── KPI content ────────────────────────────────────────────────────────
    metric_text = fields.Text(
        string="Metric Text",
        help="The representative sentence or extracted metric value",
    )
    metric_value = fields.Float(
        string="Metric Value",
        help="Numeric value if extractable (e.g. 40000 for '40,000 ha')",
        default=0.0,
    )

    # ── Qdrant bridge ─────────────────────────────────────────────────────
    # Stores a JSON list of Qdrant point UUIDs for the evidencing sentences.
    # This is the bridge between Odoo (structured) and Qdrant (semantic).
    # Flask portal reads this field, then fetches sentences from Qdrant by ID.
    qdrant_sentence_ids = fields.Text(
        string="Qdrant Sentence IDs",
        help="JSON list of Qdrant kma_sentences point UUIDs",
        default="[]",
    )
    sentence_count = fields.Integer(
        string="Sentence Count",
        compute="_compute_sentence_count",
        store=True,
    )

    # ── Review tracking ────────────────────────────────────────────────────
    reviewed      = fields.Boolean(string="Reviewed", default=False)
    reviewed_by   = fields.Many2one("res.users", string="Reviewed By")
    reviewed_date = fields.Datetime(string="Reviewed Date")
    notes         = fields.Text(string="Reviewer Notes")

    # ── Display ────────────────────────────────────────────────────────────
    name = fields.Char(
        string="Name",
        compute="_compute_name",
        store=True,
    )

    @api.depends("level_label", "class_label", "policy_id")
    def _compute_name(self):
        for rec in self:
            rec.name = (
                f"{rec.policy_id.name or ''} — "
                f"{rec.level_label or ''} / {rec.class_label or ''}"
            )

    @api.depends("qdrant_sentence_ids")
    def _compute_sentence_count(self):
        for rec in self:
            try:
                ids = json.loads(rec.qdrant_sentence_ids or "[]")
                rec.sentence_count = len(ids)
            except Exception:
                rec.sentence_count = 0

    def get_qdrant_ids(self) -> list:
        """Helper to get Qdrant IDs as a Python list."""
        try:
            return json.loads(self.qdrant_sentence_ids or "[]")
        except Exception:
            return []

    def mark_reviewed(self):
        """Button action: mark this KPI as reviewed."""
        self.write({
            "reviewed":      True,
            "reviewed_by":   self.env.user.id,
            "reviewed_date": fields.Datetime.now(),
        })

    def action_view_sentences(self):
        """
        Opens a URL to your Flask portal showing sentences for this KPI.
        Update PORTAL_URL to match your Flask server.
        """
        PORTAL_URL = "http://your-flask-portal/kpi"
        return {
            "type":   "ir.actions.act_url",
            "url":    f"{PORTAL_URL}/{self.id}/sentences",
            "target": "new",
        }


# ── Add One2many to existing aspect.policies model ────────────────────────────
class AspectPoliciesKmaExtension(models.Model):
    """
    Extends your existing aspect.policies model to add the kpi_ids field.
    No changes needed to your existing aspectPolicies class.
    """
    _inherit = "aspect.policies"

    kpi_ids = fields.One2many(
        "kma.kpi",
        "policy_id",
        string="KPIs",
    )
    kpi_count = fields.Integer(
        string="KPI Count",
        compute="_compute_kpi_count",
    )

    @api.depends("kpi_ids")
    def _compute_kpi_count(self):
        for rec in self:
            rec.kpi_count = len(rec.kpi_ids)

    def action_view_kpis(self):
        """Button to view KPIs for this policy."""
        return {
            "type":      "ir.actions.act_window",
            "name":      f"KPIs — {self.name}",
            "res_model": "kma.kpi",
            "view_mode": "list,form",
            "domain":    [("policy_id", "=", self.id)],
            "context":   {"default_policy_id": self.id},
        }
