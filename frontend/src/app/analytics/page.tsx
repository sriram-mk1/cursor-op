"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { BarChart3, TrendingUp, Users, Zap, Clock, ArrowUpRight, Activity } from "lucide-react";
import { motion } from "framer-motion";

export default function AnalyticsPage() {
    const [stats, setStats] = useState<any>(null);
    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const API_KEY = typeof window !== 'undefined' ? (localStorage.getItem("v1_key") || "v1-test-key") : "v1-test-key";

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const resp = await fetch(`${GATEWAY_URL}/api/stats`, {
                    headers: { "x-v1-key": API_KEY }
                });
                const data = await resp.json();
                if (data && typeof data === 'object' && !data.detail) {
                    setStats(data);
                } else {
                    console.error("Invalid stats data:", data);
                    setStats(null);
                }
            } catch (e) {
                console.error("Failed to fetch stats", e);
                setStats(null);
            }
        };
        fetchStats();
    }, [GATEWAY_URL, API_KEY]);

    return (
        <div className="layout-wrapper">
            <Sidebar />

            <main className="main-content">
                <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "48px" }}>
                    <div>
                        <h1 style={{ fontSize: "24px", marginBottom: "4px" }}>Analytics</h1>
                        <p className="text-muted" style={{ fontSize: "14px" }}>Deep insights into your context optimization and cost savings.</p>
                    </div>
                    <div style={{ display: "flex", gap: "12px" }}>
                        <select className="input" style={{ fontSize: "12px", padding: "6px 12px" }}>
                            <option>Last 24 Hours</option>
                            <option>Last 7 Days</option>
                            <option>Last 30 Days</option>
                        </select>
                    </div>
                </header>

                <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "24px", marginBottom: "24px" }}>
                    <div className="glass card" style={{ height: "400px", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
                        <Activity size={48} className="text-accent" style={{ opacity: 0.2, marginBottom: "16px" }} />
                        <p className="text-muted" style={{ fontSize: "14px" }}>
                            {stats?.recent_requests?.length > 0 ? "Analyzing request patterns..." : "No data available for the selected period."}
                        </p>
                    </div>

                    <div style={{ display: "grid", gap: "24px" }}>
                        <div className="glass card">
                            <div className="stat-label">Total Tokens Saved</div>
                            <div className="stat-value text-accent" style={{ fontSize: "24px", marginTop: "8px" }}>
                                {stats?.total_tokens_saved?.toLocaleString() || "0"}
                            </div>
                            <div style={{ fontSize: "11px", opacity: 0.4, marginTop: "4px" }}>Lifetime savings</div>
                        </div>
                        <div className="glass card">
                            <div className="stat-label">Total Requests</div>
                            <div className="stat-value" style={{ fontSize: "24px", marginTop: "8px" }}>
                                {stats?.total_requests?.toLocaleString() || "0"}
                            </div>
                            <div style={{ fontSize: "11px", opacity: 0.4, marginTop: "4px" }}>Processed by V1</div>
                        </div>
                        <div className="glass card">
                            <div className="stat-label">Active Sessions</div>
                            <div className="stat-value" style={{ fontSize: "24px", marginTop: "8px" }}>
                                {new Set(stats?.recent_requests?.map((r: any) => r.session_id)).size || "0"}
                            </div>
                            <div style={{ fontSize: "11px", opacity: 0.4, marginTop: "4px" }}>In the last 50 requests</div>
                        </div>
                    </div>
                </div>

                <div className="glass card">
                    <h3 style={{ fontSize: "16px", marginBottom: "20px" }}>Efficiency Breakdown</h3>
                    <div style={{ display: "grid", gap: "16px" }}>
                        {stats?.recent_requests?.length > 0 ? (
                            stats.recent_requests.slice(0, 5).map((req: any, i: number) => (
                                <div key={req.id} style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                                    <div style={{ flex: 1 }}>
                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "6px" }}>
                                            <span>Session: {req.session_id}</span>
                                            <span className="text-accent">{((req.tokens_saved / (req.tokens_in + 1)) * 100).toFixed(1)}% saved</span>
                                        </div>
                                        <div style={{ height: "4px", background: "var(--border)", borderRadius: "2px", overflow: "hidden" }}>
                                            <div style={{ width: `${Math.min(100, (req.tokens_saved / (req.tokens_in + 1)) * 100)}%`, height: "100%", background: "var(--accent)" }} />
                                        </div>
                                    </div>
                                    <div className="text-muted" style={{ fontSize: "12px", width: "80px", textAlign: "right" }}>{req.tokens_in} tkn</div>
                                </div>
                            ))
                        ) : (
                            <p className="text-muted" style={{ fontSize: "13px", textAlign: "center", padding: "20px" }}>
                                Start sending requests to see efficiency metrics.
                            </p>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
