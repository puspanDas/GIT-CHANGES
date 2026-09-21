"""
AI Product Strategist Service
Generates PRDs from feature ideas, maps to OKRs, decomposes into tasks,
and simulates A/B test validation data for PM interview showcase.
"""
import json_storage
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# ==================== OKR TEMPLATES ====================
DEFAULT_OKRS = [
    {
        "id": "okr-1",
        "objective": "Increase User Retention by 15%",
        "quarter": "Q3 2026",
        "key_results": [
            "Reduce churn rate from 8% to 5%",
            "Increase DAU/MAU ratio from 0.3 to 0.45",
            "Improve NPS score from 32 to 50"
        ],
        "category": "Growth",
        "color": "#6366f1"
    },
    {
        "id": "okr-2",
        "objective": "Improve Developer Productivity by 25%",
        "quarter": "Q3 2026",
        "key_results": [
            "Reduce average task completion time by 20%",
            "Increase sprint velocity from 40 to 50 story points",
            "Decrease bug reopen rate from 15% to 5%"
        ],
        "category": "Engineering",
        "color": "#8b5cf6"
    },
    {
        "id": "okr-3",
        "objective": "Scale Platform to 10,000 Active Users",
        "quarter": "Q4 2026",
        "key_results": [
            "Achieve 99.9% uptime SLA",
            "Reduce API p95 latency to under 200ms",
            "Support 500 concurrent WebSocket connections"
        ],
        "category": "Infrastructure",
        "color": "#06b6d4"
    },
    {
        "id": "okr-4",
        "objective": "Launch Enterprise Tier and Generate $50K MRR",
        "quarter": "Q4 2026",
        "key_results": [
            "Onboard 5 enterprise customers",
            "Achieve $50K monthly recurring revenue",
            "Maintain customer acquisition cost below $500"
        ],
        "category": "Revenue",
        "color": "#10b981"
    },
    {
        "id": "okr-5",
        "objective": "Achieve Best-in-Class User Experience",
        "quarter": "Q3 2026",
        "key_results": [
            "Increase task completion rate to 90%",
            "Reduce time-to-first-action below 30 seconds",
            "Achieve 4.5+ star rating on G2/Capterra"
        ],
        "category": "Product",
        "color": "#f59e0b"
    }
]

# ==================== PRD GENERATION ENGINE ====================

# Domain knowledge patterns for generating realistic PRDs
PRD_PATTERNS = {
    "dark mode": {
        "problem": "Users report eye strain during extended evening work sessions, and 73% of surveyed users prefer dark interfaces in productivity tools. The current light-only theme is causing measurable churn among power users who spend 4+ hours daily on the platform.",
        "personas": [
            {"name": "Night Owl Developer", "description": "Works late shifts, sensitive to bright screens, uses dark mode in all other tools (VS Code, Slack, GitHub)"},
            {"name": "Accessibility-Focused PM", "description": "Has a visual sensitivity condition, needs low-contrast options to work comfortably"}
        ],
        "success_metrics": ["40% adoption within 2 weeks of launch", "15% reduction in evening session churn", "Positive sentiment in 80% of user feedback"],
        "out_of_scope": ["Custom theme builder", "Per-component theme overrides", "Theme marketplace"],
        "edge_cases": ["Charts and data visualizations must remain readable", "User-uploaded images should not be inverted", "Email notifications should retain brand colors"],
        "risks": ["Inconsistent contrast ratios may fail WCAG 2.1 AA compliance", "Third-party embedded widgets may not respect theme"],
        "estimated_points": 13
    },
    "sso": {
        "problem": "Enterprise prospects consistently cite the lack of Single Sign-On as a dealbreaker during sales calls. 4 out of 5 lost enterprise deals in Q2 mentioned SSO as a required feature. Manual account provisioning creates security risks and increases onboarding friction.",
        "personas": [
            {"name": "IT Administrator", "description": "Manages 500+ employee accounts, needs centralized identity management and audit trails"},
            {"name": "Enterprise CISO", "description": "Requires SAML 2.0/OIDC compliance for security policy adherence"}
        ],
        "success_metrics": ["Close 3 enterprise deals blocked by SSO requirement", "Reduce enterprise onboarding time from 2 weeks to 1 day", "Zero SSO-related security incidents post-launch"],
        "out_of_scope": ["Multi-factor authentication (separate initiative)", "Custom identity provider development", "Legacy LDAP support"],
        "edge_cases": ["User has both SSO and local account with same email", "SSO provider is temporarily unavailable", "User is removed from IdP but has active sessions"],
        "risks": ["SAML XML signature wrapping attacks", "Token replay vulnerabilities if not implementing proper nonce validation"],
        "estimated_points": 21
    },
    "notification": {
        "problem": "Users miss critical task updates because the platform lacks a real-time notification system. 62% of overdue tasks were caused by assignees not being aware of status changes or new comments. This directly impacts sprint velocity.",
        "personas": [
            {"name": "Busy Developer", "description": "Juggles 5-8 tasks simultaneously, needs push alerts for blockers and review requests"},
            {"name": "Remote PM", "description": "Manages distributed team across timezones, needs digest summaries of overnight activity"}
        ],
        "success_metrics": ["Reduce average task response time from 4 hours to 30 minutes", "80% of users enable at least one notification channel", "Decrease overdue tasks by 25%"],
        "out_of_scope": ["SMS notifications", "Third-party webhook integrations", "Custom notification sounds"],
        "edge_cases": ["User is mentioned in a comment on a task they're not assigned to", "Notification sent for a task that gets deleted before user reads it", "Bulk operations triggering notification storms"],
        "risks": ["Push notification fatigue leading to users disabling all notifications", "GDPR compliance for email notification preferences"],
        "estimated_points": 13
    },
    "mobile": {
        "problem": "35% of user sessions start on mobile devices but the current web interface is not optimized for small screens. Mobile users complete 60% fewer tasks compared to desktop users, indicating a significant UX gap that's limiting platform adoption.",
        "personas": [
            {"name": "On-the-Go PM", "description": "Needs to approve tasks, check sprint progress, and respond to blockers during commute or meetings"},
            {"name": "Field Engineer", "description": "Reports bugs and updates task status from customer sites using a phone"}
        ],
        "success_metrics": ["Achieve feature parity for core actions (view, update, comment) on mobile", "Increase mobile task completion rate from 15% to 60%", "Maintain sub-3-second load time on 4G connections"],
        "out_of_scope": ["Native iOS/Android apps (PWA first)", "Offline mode", "Mobile-specific features not on desktop"],
        "edge_cases": ["Drag-and-drop Kanban board on touch devices", "File upload from mobile camera", "Long task descriptions on small screens"],
        "risks": ["Touch target sizes below 44px causing accessibility issues", "Mobile Safari viewport height bugs with fixed headers"],
        "estimated_points": 21
    },
    "report": {
        "problem": "Product managers spend an average of 3 hours per week manually compiling sprint reports from exported CSV data. Stakeholders need automated, visually compelling reports to make data-driven decisions in leadership meetings.",
        "personas": [
            {"name": "VP of Engineering", "description": "Needs weekly executive summary of sprint progress, velocity trends, and risk areas—delivered automatically"},
            {"name": "Scrum Master", "description": "Requires detailed burndown charts, cycle time analysis, and team workload distribution for retrospectives"}
        ],
        "success_metrics": ["Reduce report generation time from 3 hours to 5 minutes", "100% of PMs generate at least one automated report per sprint", "Stakeholder satisfaction with report quality above 4/5"],
        "out_of_scope": ["Custom report designer/builder", "Real-time dashboard streaming", "Integration with BI tools (Tableau, PowerBI)"],
        "edge_cases": ["Sprint with zero completed tasks", "Team member added mid-sprint skewing velocity", "Tasks reassigned between team members during sprint"],
        "risks": ["Misleading velocity trends from small sample sizes", "Report data staleness if caching is too aggressive"],
        "estimated_points": 13
    }
}

# Generic fallback for unrecognized ideas
GENERIC_PRD = {
    "problem": "This feature addresses a key gap in the current product offering. Based on competitive analysis and user feedback, implementing this capability will improve user satisfaction and differentiate the platform in the market.",
    "personas": [
        {"name": "Power User", "description": "Daily active user who relies on the platform for core workflows and expects advanced capabilities"},
        {"name": "Team Lead", "description": "Manages a team of 5-10 and needs features that improve team coordination and visibility"}
    ],
    "success_metrics": ["30% feature adoption within first month", "Positive user feedback (>4/5 satisfaction)", "No increase in support ticket volume"],
    "out_of_scope": ["Enterprise-specific customizations", "Third-party integrations", "Admin configuration UI"],
    "edge_cases": ["Concurrent usage by multiple team members", "Edge case with empty data states", "Permission boundaries between roles"],
    "risks": ["Scope creep during implementation", "Potential performance impact on existing features"],
    "estimated_points": 8
}


def _match_pattern(idea: str) -> Dict:
    """Match a feature idea to a known pattern for realistic PRD generation."""
    idea_lower = idea.lower()
    for key, pattern in PRD_PATTERNS.items():
        if key in idea_lower:
            return pattern
    # Check for partial keyword matches
    keyword_map = {
        "dark": "dark mode", "theme": "dark mode", "night": "dark mode",
        "sso": "sso", "login": "sso", "oauth": "sso", "saml": "sso", "sign-on": "sso",
        "notify": "notification", "alert": "notification", "push": "notification", "email": "notification",
        "mobile": "mobile", "responsive": "mobile", "phone": "mobile", "app": "mobile",
        "report": "report", "dashboard": "report", "analytics": "report", "chart": "report", "export": "report"
    }
    for keyword, pattern_key in keyword_map.items():
        if keyword in idea_lower:
            return PRD_PATTERNS[pattern_key]
    return GENERIC_PRD


def generate_prd(idea: str, okr_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a structured Product Requirements Document from a 1-sentence feature idea.
    Returns a complete PRD with problem statement, personas, metrics, scope, and risks.
    """
    pattern = _match_pattern(idea)
    
    # Find matching OKR
    matched_okr = None
    if okr_id:
        matched_okr = next((o for o in DEFAULT_OKRS if o["id"] == okr_id), None)
    
    # Auto-suggest OKR if none provided
    if not matched_okr:
        idea_lower = idea.lower()
        if any(w in idea_lower for w in ["retain", "churn", "engage", "dark", "theme", "ux"]):
            matched_okr = DEFAULT_OKRS[0]  # Retention
        elif any(w in idea_lower for w in ["speed", "fast", "automate", "sprint", "velocity"]):
            matched_okr = DEFAULT_OKRS[1]  # Dev Productivity
        elif any(w in idea_lower for w in ["scale", "performance", "infra", "api", "load"]):
            matched_okr = DEFAULT_OKRS[2]  # Scale
        elif any(w in idea_lower for w in ["enterprise", "sso", "revenue", "pricing", "tier"]):
            matched_okr = DEFAULT_OKRS[3]  # Revenue
        else:
            matched_okr = DEFAULT_OKRS[4]  # UX (default)

    prd = {
        "id": _generate_id(idea),
        "title": idea.strip(),
        "status": "DRAFT",
        "created_at": datetime.now().isoformat(),
        "author": "Product Manager",
        "okr": matched_okr,
        "problem_statement": pattern["problem"],
        "target_personas": pattern["personas"],
        "success_metrics": pattern["success_metrics"],
        "out_of_scope": pattern["out_of_scope"],
        "edge_cases": pattern["edge_cases"],
        "risks": pattern["risks"],
        "estimated_story_points": pattern["estimated_points"],
        "priority_recommendation": _calculate_priority(pattern),
        "competitive_analysis": _generate_competitive_analysis(idea),
        "implementation_phases": _generate_phases(idea, pattern)
    }
    
    return prd


def decompose_prd_to_tasks(prd: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Break a PRD down into actionable engineering tasks with story point estimates.
    Returns a list of task objects ready to be created in the system.
    """
    idea = prd.get("title", "").lower()
    phases = prd.get("implementation_phases", [])
    
    tasks = []
    task_templates = _get_task_templates(idea)
    
    for i, template in enumerate(task_templates):
        phase_idx = min(i // 2, len(phases) - 1) if phases else 0
        phase_name = phases[phase_idx]["name"] if phases and phase_idx < len(phases) else "Phase 1"
        
        tasks.append({
            "title": template["title"],
            "description": template["description"],
            "priority": template["priority"],
            "estimated_days": template["days"],
            "story_points": template["points"],
            "phase": phase_name,
            "type": template["type"],
            "acceptance_criteria": template.get("acceptance_criteria", [])
        })
    
    return tasks


def get_okrs() -> List[Dict[str, Any]]:
    """Return all available OKRs."""
    return DEFAULT_OKRS


def simulate_ab_test(prd_id: str, feature_name: str) -> Dict[str, Any]:
    """
    Simulate A/B test results for a shipped feature.
    Generates realistic mock data for interview demonstration.
    """
    # Use feature name as seed for consistent results per feature
    seed = int(hashlib.md5(feature_name.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    
    # Generate realistic metrics
    control_conversion = round(rng.uniform(2.0, 8.0), 2)
    variant_lift = round(rng.uniform(-3.0, 12.0), 2)
    variant_conversion = round(control_conversion + (control_conversion * variant_lift / 100), 2)
    
    sample_size = rng.randint(5000, 50000)
    confidence = round(rng.uniform(85.0, 99.5), 1)
    is_significant = confidence >= 95.0
    
    adoption_rate = round(rng.uniform(15.0, 72.0), 1)
    
    # Daily adoption curve (simulates gradual rollout)
    daily_data = []
    base = rng.uniform(5, 15)
    for day in range(14):
        growth = base + (adoption_rate - base) * (1 - 0.85 ** (day + 1))
        daily_data.append({
            "day": day + 1,
            "date": (datetime.now() - timedelta(days=13 - day)).strftime("%b %d"),
            "adoption_pct": round(growth + rng.uniform(-2, 2), 1),
            "control_conversion": round(control_conversion + rng.uniform(-0.5, 0.5), 2),
            "variant_conversion": round(variant_conversion + rng.uniform(-0.5, 0.5), 2)
        })
    
    # User segments
    segments = [
        {
            "name": "Power Users (daily active)",
            "adoption": round(adoption_rate * rng.uniform(1.2, 1.8), 1),
            "satisfaction": round(rng.uniform(3.8, 4.9), 1)
        },
        {
            "name": "Casual Users (weekly active)",
            "adoption": round(adoption_rate * rng.uniform(0.4, 0.8), 1),
            "satisfaction": round(rng.uniform(3.2, 4.5), 1)
        },
        {
            "name": "New Users (< 30 days)",
            "adoption": round(adoption_rate * rng.uniform(0.6, 1.1), 1),
            "satisfaction": round(rng.uniform(3.5, 4.7), 1)
        }
    ]
    
    # Generate recommendation
    if is_significant and variant_lift > 0:
        recommendation = "SHIP_IT"
        recommendation_text = f"The variant shows a statistically significant +{variant_lift}% lift in conversion. Recommend full rollout."
    elif is_significant and variant_lift < 0:
        recommendation = "ITERATE"
        recommendation_text = f"The variant shows a significant {variant_lift}% decrease. Recommend iterating on the design before re-testing."
    else:
        recommendation = "EXTEND_TEST"
        recommendation_text = f"Results are not yet statistically significant at {confidence}% confidence. Recommend extending the test for 1-2 more weeks."

    return {
        "prd_id": prd_id,
        "feature_name": feature_name,
        "test_duration_days": 14,
        "sample_size": sample_size,
        "control": {
            "name": "Control (No Feature)",
            "conversion_rate": control_conversion,
            "users": sample_size // 2
        },
        "variant": {
            "name": f"Variant ({feature_name})",
            "conversion_rate": variant_conversion,
            "users": sample_size - sample_size // 2
        },
        "lift_pct": variant_lift,
        "confidence_pct": confidence,
        "is_significant": is_significant,
        "adoption_rate": adoption_rate,
        "daily_data": daily_data,
        "segments": segments,
        "recommendation": recommendation,
        "recommendation_text": recommendation_text,
        "secondary_metrics": {
            "avg_session_duration_change": f"{'+' if rng.random() > 0.3 else '-'}{round(rng.uniform(1, 15), 1)}%",
            "support_tickets_change": f"{'+' if rng.random() > 0.6 else '-'}{round(rng.uniform(1, 20), 1)}%",
            "page_load_impact_ms": round(rng.uniform(-50, 150), 0)
        }
    }


# ==================== HELPER FUNCTIONS ====================

def _generate_id(idea: str) -> str:
    """Generate a deterministic ID from the idea text."""
    hash_hex = hashlib.md5(idea.strip().lower().encode()).hexdigest()[:8]
    return f"prd-{hash_hex}"


def _calculate_priority(pattern: Dict) -> str:
    """Calculate priority recommendation based on pattern data."""
    points = pattern.get("estimated_points", 8)
    if points >= 21:
        return "HIGH"
    elif points >= 13:
        return "MEDIUM"
    return "LOW"


def _generate_competitive_analysis(idea: str) -> List[Dict[str, str]]:
    """Generate a competitive analysis table."""
    idea_lower = idea.lower()
    
    competitors = [
        {"name": "Jira", "has_feature": True, "quality": "Basic", "notes": "Available but buried in settings, requires admin configuration"},
        {"name": "Asana", "has_feature": True, "quality": "Good", "notes": "Well-implemented but limited customization options"},
        {"name": "Linear", "has_feature": True, "quality": "Excellent", "notes": "Best-in-class implementation with smooth UX"},
        {"name": "Monday.com", "has_feature": False, "quality": "N/A", "notes": "Not available, users rely on workarounds"},
        {"name": "TaskFlow (Us)", "has_feature": False, "quality": "Planned", "notes": "Will differentiate with AI-powered approach"}
    ]
    
    # Customize based on idea
    if any(w in idea_lower for w in ["dark", "theme"]):
        competitors[0]["notes"] = "Dark mode available since 2020, basic implementation"
        competitors[2]["quality"] = "Excellent"
        competitors[2]["notes"] = "Native dark mode, auto-detects system preference"
        competitors[4]["notes"] = "Will include auto-detect + scheduled switching"
    elif any(w in idea_lower for w in ["sso", "oauth"]):
        competitors[0]["has_feature"] = True
        competitors[0]["quality"] = "Enterprise"
        competitors[0]["notes"] = "Full SAML/OIDC but only on Premium tier ($$$)"
        competitors[3]["has_feature"] = True
        competitors[3]["quality"] = "Good"
        competitors[3]["notes"] = "Available on Enterprise plan"
        competitors[4]["notes"] = "Will offer SSO on all paid tiers—competitive advantage"
    
    return competitors


def _generate_phases(idea: str, pattern: Dict) -> List[Dict[str, Any]]:
    """Generate implementation phases for the PRD."""
    total_points = pattern.get("estimated_points", 8)
    
    if total_points >= 21:
        return [
            {"name": "Phase 1: Foundation", "duration": "Sprint 1 (2 weeks)", "scope": "Core infrastructure and data models", "points": 8},
            {"name": "Phase 2: Core Features", "duration": "Sprint 2 (2 weeks)", "scope": "Primary user-facing functionality", "points": 8},
            {"name": "Phase 3: Polish & Launch", "duration": "Sprint 3 (1 week)", "scope": "Edge cases, testing, and rollout", "points": 5}
        ]
    elif total_points >= 13:
        return [
            {"name": "Phase 1: MVP", "duration": "Sprint 1 (2 weeks)", "scope": "Core functionality and basic UI", "points": 8},
            {"name": "Phase 2: Refinement", "duration": "Sprint 2 (1 week)", "scope": "Edge cases, polish, and launch", "points": 5}
        ]
    else:
        return [
            {"name": "Phase 1: Implementation", "duration": "1 Sprint (2 weeks)", "scope": "Full feature implementation and testing", "points": total_points}
        ]


def _get_task_templates(idea: str) -> List[Dict[str, Any]]:
    """Get task breakdown templates based on the feature idea."""
    idea_lower = idea.lower()
    
    if any(w in idea_lower for w in ["dark", "theme", "night"]):
        return [
            {"title": "Define CSS custom properties for dark theme", "description": "Create a comprehensive set of CSS custom properties (--color-bg, --color-text, --color-surface, etc.) that will power the theme switching. Audit all existing hardcoded colors.", "priority": "HIGH", "days": 2, "points": 3, "type": "Frontend", "acceptance_criteria": ["All colors use CSS variables", "No hardcoded hex values remain"]},
            {"title": "Implement ThemeProvider context and toggle", "description": "Create a React Context provider that manages theme state, persists preference to localStorage, and respects system prefers-color-scheme.", "priority": "HIGH", "days": 1.5, "points": 2, "type": "Frontend", "acceptance_criteria": ["Theme persists across sessions", "Respects OS setting on first visit"]},
            {"title": "Update all components to use theme tokens", "description": "Systematically update Dashboard, KanbanBoard, ListView, TaskDetail, and all other components to use theme-aware CSS classes instead of hardcoded colors.", "priority": "MEDIUM", "days": 3, "points": 5, "type": "Frontend", "acceptance_criteria": ["All components render correctly in both themes", "No visual regressions"]},
            {"title": "Add dark mode variants for charts and graphs", "description": "Update Analytics, SprintPlanner, and DependencyGraph visualizations to be theme-aware with proper contrast ratios.", "priority": "MEDIUM", "days": 1.5, "points": 2, "type": "Frontend", "acceptance_criteria": ["Charts readable in dark mode", "WCAG AA contrast compliance"]},
            {"title": "Write accessibility audit tests", "description": "Test all WCAG 2.1 AA contrast ratios in both themes. Verify focus indicators are visible. Test with screen readers.", "priority": "LOW", "days": 1, "points": 1, "type": "QA"}
        ]
    elif any(w in idea_lower for w in ["sso", "oauth", "login", "sign-on"]):
        return [
            {"title": "Design SSO authentication flow and data model", "description": "Define the SAML 2.0 and OIDC authentication flows, session management strategy, and user provisioning/de-provisioning data model.", "priority": "HIGH", "days": 1, "points": 2, "type": "Architecture"},
            {"title": "Implement SAML 2.0 service provider", "description": "Add SAML SP endpoints: metadata, ACS (Assertion Consumer Service), SLO. Parse SAML assertions and map attributes to user model.", "priority": "CRITICAL", "days": 4, "points": 8, "type": "Backend", "acceptance_criteria": ["SAML metadata endpoint serves valid XML", "ACS processes assertions correctly", "SLO terminates sessions"]},
            {"title": "Implement OIDC/OAuth2 integration", "description": "Add OIDC client for Google Workspace, Azure AD, and Okta. Implement authorization code flow with PKCE.", "priority": "HIGH", "days": 3, "points": 5, "type": "Backend", "acceptance_criteria": ["Google Workspace login works", "Azure AD login works", "Token refresh handled"]},
            {"title": "Build SSO configuration admin panel", "description": "Create an admin UI where IT admins can configure their IdP settings, upload metadata XML, and test the connection.", "priority": "MEDIUM", "days": 2, "points": 3, "type": "Frontend", "acceptance_criteria": ["Admin can paste IdP metadata", "Test connection button validates setup"]},
            {"title": "Add JIT user provisioning and role mapping", "description": "When a user logs in via SSO for the first time, auto-create their account and map IdP groups to TaskFlow roles.", "priority": "MEDIUM", "days": 1.5, "points": 2, "type": "Backend"},
            {"title": "Security testing and penetration test", "description": "Test for SAML signature wrapping, token replay, open redirect, and session fixation vulnerabilities.", "priority": "HIGH", "days": 1, "points": 1, "type": "QA"}
        ]
    elif any(w in idea_lower for w in ["notification", "alert", "push"]):
        return [
            {"title": "Design notification data model and preferences schema", "description": "Define notification types (mention, assignment, status_change, comment), delivery channels (in-app, email, push), and user preference storage.", "priority": "HIGH", "days": 1, "points": 2, "type": "Architecture"},
            {"title": "Build notification service backend", "description": "Create notification_service.py with event listeners for task updates, comments, and mentions. Queue notifications for delivery.", "priority": "HIGH", "days": 2, "points": 3, "type": "Backend", "acceptance_criteria": ["Events trigger notifications", "Deduplication prevents spam"]},
            {"title": "Implement in-app notification bell and panel", "description": "Add a notification bell icon in the header with unread count badge. Build a dropdown panel showing recent notifications with mark-as-read.", "priority": "HIGH", "days": 2, "points": 3, "type": "Frontend", "acceptance_criteria": ["Bell shows unread count", "Panel lists last 50 notifications", "Mark as read works"]},
            {"title": "Add real-time notification delivery via WebSocket", "description": "Extend existing WebSocket infrastructure to push notifications instantly. Show toast alerts for high-priority notifications.", "priority": "MEDIUM", "days": 1.5, "points": 3, "type": "Full Stack"},
            {"title": "Build notification preferences page", "description": "Create a settings page where users control which events trigger notifications and through which channels.", "priority": "LOW", "days": 1, "points": 2, "type": "Frontend"}
        ]
    else:
        # Generic task decomposition
        return [
            {"title": f"Technical spike: Research and design {idea[:50]}", "description": f"Conduct a technical investigation into implementing '{idea}'. Evaluate approaches, identify risks, and propose architecture. Produce a design document.", "priority": "HIGH", "days": 1, "points": 2, "type": "Architecture"},
            {"title": f"Backend: Core API and data model for {idea[:40]}", "description": f"Implement the backend data model, storage functions in json_storage.py, and REST API endpoints for the feature.", "priority": "HIGH", "days": 2, "points": 3, "type": "Backend", "acceptance_criteria": ["API endpoints return correct responses", "Data persists across restarts"]},
            {"title": f"Frontend: Build UI components for {idea[:40]}", "description": f"Create the React components needed for the feature. Implement the main view, any modals, and integrate with the API.", "priority": "HIGH", "days": 3, "points": 5, "type": "Frontend", "acceptance_criteria": ["Components render correctly", "API integration works", "Responsive on desktop"]},
            {"title": f"Integration testing and edge cases", "description": "Write integration tests covering the happy path and edge cases. Test error handling, empty states, and concurrent access.", "priority": "MEDIUM", "days": 1, "points": 2, "type": "QA"},
            {"title": f"Documentation and release notes", "description": "Update user documentation, add feature flag configuration, and write release notes for the changelog.", "priority": "LOW", "days": 0.5, "points": 1, "type": "Documentation"}
        ]


# ==================== STORAGE FUNCTIONS ====================

def save_prd(prd: Dict[str, Any]) -> Dict[str, Any]:
    """Save a PRD to the data store."""
    data = json_storage.load_data()
    if "prds" not in data:
        data["prds"] = []
    
    # Check if PRD with same ID exists, update it
    existing_idx = next((i for i, p in enumerate(data["prds"]) if p["id"] == prd["id"]), None)
    if existing_idx is not None:
        data["prds"][existing_idx] = prd
    else:
        data["prds"].append(prd)
    
    json_storage.save_data(data)
    return prd


def get_all_prds() -> List[Dict[str, Any]]:
    """Get all saved PRDs."""
    data = json_storage.load_data()
    return data.get("prds", [])


def get_prd_by_id(prd_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific PRD by ID."""
    data = json_storage.load_data()
    prds = data.get("prds", [])
    return next((p for p in prds if p["id"] == prd_id), None)


def update_prd_status(prd_id: str, new_status: str) -> Optional[Dict[str, Any]]:
    """Update the status of a PRD (DRAFT -> APPROVED -> IN_PROGRESS -> SHIPPED)."""
    data = json_storage.load_data()
    if "prds" not in data:
        return None
    
    for prd in data["prds"]:
        if prd["id"] == prd_id:
            prd["status"] = new_status
            prd["updated_at"] = datetime.now().isoformat()
            json_storage.save_data(data)
            return prd
    return None


def delete_prd(prd_id: str) -> bool:
    """Delete a PRD."""
    data = json_storage.load_data()
    if "prds" not in data:
        return False
    
    original_len = len(data["prds"])
    data["prds"] = [p for p in data["prds"] if p["id"] != prd_id]
    
    if len(data["prds"]) < original_len:
        json_storage.save_data(data)
        return True
    return False
