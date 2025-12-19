"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { BarChart3, TrendingUp, Users, Zap, Clock, ArrowUpRight, Activity } from "lucide-react";
import { motion } from "framer-motion";

export default function AnalyticsPage() {
    const [stats, setStats] = useState<any>(null);
    const [view, setView] = useState("daily");
    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const API_KEY = typeof window !== 'undefined' ? (localStorage.getItem("v1_key") || "") : "";

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const resp = await fetch(`${GATEWAY_URL}/api/stats`, {
                    headers: { "x-v1-key": API_KEY }
                });
                const data = await resp.json();
                if (data && typeof data === 'object' && !data.detail) {
                    setStats(data);
                }
            } catch (e) {
                console.error("Failed to fetch stats", e);
            }
        };
        fetchStats();
    }, [GATEWAY_URL, API_KEY]);

    const recentLogs = stats?.recent_requests || [];
    const totalSaved = recentLogs.reduce((acc: number, r: any) => acc + (r.cost_saved_usd || 0), 0);
    const avgLatency = recentLogs.length > 0 ? recentLogs.reduce((acc: number, r: any) => acc + r.latency_ms, 0) / recentLogs.length : 0;
    const avgOptimization = recentLogs.length > 0 ? (recentLogs.reduce((acc: number, r: any) => acc + r.tokens_saved, 0) / recentLogs.reduce((acc: number, r: any) => acc + (r.tokens_in + r.tokens_saved), 0)) * 100 : 0;

    return (
        <div className="layout-wrapper">
            <Sidebar />

            <main className="main-content">
                <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "48px" }}>
                    <div>
                        <h1 style={{ fontSize: "24px", fontWeight: 600 }}>Performance Analytics</h1>
                        <p className="text-muted" style={{ fontSize: "13px", marginTop: "4px" }}>Visualizing the impact of V1 Context Optimization.</p>
                    </div>
                    <div className="glass" style={{ padding: "4px", borderRadius: "8px", display: "flex", gap: "4px" }}>
                        {["daily", "weekly", "monthly"].map(v => (
                            <button
                                key={v}
                                onClick={() => setView(v)}
                                className="btn"
                                style={{
                                    background: view === v ? "var(--fg)" : "transparent",
                                    color: view === v ? "var(--bg)" : "var(--fg)",
                                    fontSize: "10px",
                                    padding: "6px 12px",
                                    textTransform: "uppercase"
                                }}
                            >
                                {v}
                            </button>
                        ))}
                    </div>
                </header>

                <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "32px", marginBottom: "32px" }}>
                    {/* Main Chart Card */}
                    <div className="glass card" style={{ position: "relative" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "32px" }}>
                            <div>
                                <h3 style={{ fontSize: "14px", fontWeight: 600, color: "var(--muted)" }}>SAVINGS TREND</h3>
                                <div style={{ fontSize: "24px", fontWeight: 500, marginTop: "4px" }}>${totalSaved.toFixed(4)}</div>
                            </div>
                            <div className="savings-badge">
                                <TrendingUp size={12} /> +{avgOptimization.toFixed(1)}% Efficiency
                            </div>
                        </div>

                        <div className="bar-chart-container">
                            {recentLogs.length > 0 ? [...recentLogs].reverse().map((r: any, i: number) => {
                                const height = (r.cost_saved_usd / (Math.max(...recentLogs.map((l: any) => l.cost_saved_usd)) || 1)) * 100;
                                return (
                                    <motion.div
                                        key={r.id}
                                        initial={{ height: 0 }}
                                        animate={{ height: `${Math.max(10, height)}%` }}
                                        className="bar-column"
                                        data-value={`$${(r.cost_saved_usd || 0).toFixed(4)}`}
                                    />
                                );
                            }) : (
                                <div className="text-muted" style={{ width: "100%", textAlign: "center", fontSize: "12px" }}>Awaiting request data...</div>
                            )}
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "12px", fontSize: "10px", opacity: 0.3 }}>
                            <span>50 REQUESTS AGO</span>
                            <span>NOW</span>
                        </div>
                    </div>

                    {/* Secondary Stats */}
                    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                        <div className="glass card stat-item" style={{ background: "var(--accent-muted)", borderColor: "var(--accent)" }}>
                            <div className="stat-label" style={{ color: "var(--accent)" }}>Net Savings (Est)</div>
                            <div className="stat-value" style={{ color: "var(--accent)" }}>${(totalSaved * 0.95).toFixed(4)}</div>
                            <div style={{ fontSize: "11px", opacity: 0.6 }}>After gateway overhead</div>
                        </div>
                        <div className="glass card stat-item">
                            <div className="stat-label">Avg. Optimization</div>
                            <div className="stat-value">{avgOptimization.toFixed(1)}% <span style={{ fontSize: "14px", opacity: 0.4 }}>Tokens Distilled</span></div>
                        </div>
                        <div className="glass card stat-item">
                            <div className="stat-label">Avg. Latency</div>
                            <div className="stat-value">{avgLatency.toFixed(0)} <span style={{ fontSize: "14px", opacity: 0.4 }}>ms</span></div>
                        </div>
                    </div>
                </div>

                {/* Efficiency Breakdown */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px" }}>
                    <div className="glass card">
                        <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "24px" }}>Context Efficiency</h3>
                        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                            {recentLogs.slice(0, 6).map((req: any) => (
                                <div key={req.id}>
                                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "8px" }}>
                                        <span style={{ opacity: 0.5 }}>{req.session_id.slice(0, 16)}...</span>
                                        <span className="text-accent">{(req.tokens_saved / (req.tokens_in + req.tokens_saved || 1) * 100).toFixed(0)}% Saved</span>
                                    </div>
                                    <div style={{ height: "6px", background: "rgba(255,255,255,0.05)", borderRadius: "3px", overflow: "hidden" }}>
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${(req.tokens_saved / (req.tokens_in + req.tokens_saved || 1)) * 100}%` }}
                                            style={{ height: "100%", background: "var(--accent)" }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="glass card">
                        <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "24px" }}>Model Distribution</h3>
                        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                            {Array.from(new Set(recentLogs.map((r: any) => r.model))).map((model: any) => {
                                const count = recentLogs.filter((r: any) => r.model === model).length;
                                const percentage = (count / recentLogs.length) * 100;
                                return (
                                    <div key={model} className="glass" style={{ padding: "12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                                            <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent)" }} />
                                            <span style={{ fontSize: "12px" }}>{model}</span>
                                        </div>
                                        <div style={{ fontSize: "12px", fontWeight: 600 }}>{percentage.toFixed(0)}%</div>
                                    </div>
                                );
                            })}
                            {recentLogs.length === 0 && <div className="text-muted" style={{ fontSize: "12px", textAlign: "center", padding: "40px" }}>No model data yet</div>}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
