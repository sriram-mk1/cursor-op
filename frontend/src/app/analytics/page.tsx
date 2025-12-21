"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { BarChart3, TrendingUp, Zap, Clock, ExternalLink, ChevronRight, Search } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { createClient } from "@/utils/supabase";

export default function AnalyticsPage() {
    const router = useRouter();
    const [stats, setStats] = useState<any>(null);
    const [requests, setRequests] = useState<any[]>([]);
    const [view, setView] = useState("daily");
    const [loading, setLoading] = useState(true);

    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const supabase = createClient();

    useEffect(() => {
        const fetchData = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            const userId = user?.id || (process.env.NODE_ENV === 'development' ? 'dev-user' : null);

            if (userId) {
                try {
                    const resp = await fetch(`${GATEWAY_URL}/api/user/stats`, {
                        headers: { "x-user-id": userId }
                    });
                    const data = await resp.json();
                    setStats(data);

                    const fetchedReqs = data.recent_requests || [];
                    if (fetchedReqs.length === 0 && process.env.NODE_ENV === 'development') {
                        setRequests([
                            { id: 'mock-1', model: 'anthropic/claude-3-sonnet', tokens_in: 5200, tokens_saved: 4100, latency_ms: 1840, timestamp: Date.now() / 1000 - 3600 },
                            { id: 'mock-2', model: 'openai/gpt-4-turbo', tokens_in: 8400, tokens_saved: 6200, latency_ms: 3200, timestamp: Date.now() / 1000 - 7200 },
                            { id: 'mock-3', model: 'meta-llama/llama-3-70b', tokens_in: 2100, tokens_saved: 1500, latency_ms: 950, timestamp: Date.now() / 1000 - 10800 },
                        ]);
                    } else {
                        setRequests(fetchedReqs.sort((a: any, b: any) => b.timestamp - a.timestamp));
                    }
                } catch (e) {
                    console.error("Failed to fetch analytics", e);
                }
            }
            setLoading(false);
        };
        fetchData();
    }, [GATEWAY_URL, supabase.auth]);

    const recentLogs = requests;
    const totalSaved = recentLogs.reduce((acc: number, r: any) => acc + (r.cost_saved_usd || 0), 0);
    const avgLatency = recentLogs.length > 0 ? recentLogs.reduce((acc: number, r: any) => acc + r.latency_ms, 0) / recentLogs.length : 0;
    const avgOptimization = recentLogs.length > 0 ? (recentLogs.reduce((acc: number, r: any) => acc + r.tokens_saved, 0) / recentLogs.reduce((acc: number, r: any) => acc + (r.tokens_in + r.tokens_saved), 0)) * 100 : 0;

    return (
        <div className="layout-wrapper">
            <Sidebar />

            <main className="main-content">
                <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "48px" }}>
                    <div>
                        <h1 style={{ fontSize: "24px", fontWeight: 700, letterSpacing: '-0.03em' }}>Analytics Dashboard</h1>
                        <p className="text-muted" style={{ fontSize: "14px", marginTop: "4px" }}>Performance monitoring and context optimization metrics.</p>
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
                                    fontWeight: 700,
                                    textTransform: "uppercase"
                                }}
                            >
                                {v}
                            </button>
                        ))}
                    </div>
                </header>

                <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "32px", marginBottom: "48px" }}>
                    {/* Main Chart Card */}
                    <div className="glass card" style={{ position: "relative" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "32px" }}>
                            <div>
                                <h3 style={{ fontSize: "11px", fontWeight: 800, color: "var(--muted)", textTransform: 'uppercase', letterSpacing: '0.05em' }}>SAVINGS TREND</h3>
                                <div style={{ fontSize: "28px", fontWeight: 600, marginTop: "8px" }}>${totalSaved.toFixed(4)}</div>
                            </div>
                            <div className="savings-badge">
                                <TrendingUp size={12} /> {avgOptimization.toFixed(1)}% Optimized
                            </div>
                        </div>

                        <div className="bar-chart-container" style={{ height: '200px', display: 'flex', alignItems: 'flex-end', gap: '4px' }}>
                            {recentLogs.length > 0 ? [...recentLogs].reverse().map((r: any, i: number) => {
                                const height = (r.tokens_saved / (Math.max(...recentLogs.map((l: any) => l.tokens_saved)) || 1)) * 100;
                                return (
                                    <motion.div
                                        key={r.id}
                                        initial={{ height: 0 }}
                                        animate={{ height: `${Math.max(10, height)}%` }}
                                        style={{ flex: 1, background: 'var(--accent)', opacity: 0.3 + (i / recentLogs.length) * 0.7, borderRadius: '2px' }}
                                    />
                                );
                            }) : (
                                <div className="text-muted" style={{ width: "100%", textAlign: "center", fontSize: "12px" }}>Awaiting data...</div>
                            )}
                        </div>
                    </div>

                    {/* Secondary Stats */}
                    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                        <div className="glass card stat-item" style={{ background: "var(--accent-muted)", borderColor: "var(--accent)" }}>
                            <div className="stat-label" style={{ color: "var(--accent)" }}>Net Cost Savings</div>
                            <div className="stat-value" style={{ color: "var(--accent)" }}>${(totalSaved).toFixed(4)}</div>
                            <div style={{ fontSize: "11px", opacity: 0.6 }}>Cumulative across all sessions</div>
                        </div>
                        <div className="glass card stat-item">
                            <div className="stat-label">Avg. Latency</div>
                            <div className="stat-value">{avgLatency.toFixed(0)} <span style={{ fontSize: "14px", opacity: 0.4 }}>ms</span></div>
                        </div>
                        <div className="glass card stat-item">
                            <div className="stat-label">Token Savings Rate</div>
                            <div className="stat-value">{avgOptimization.toFixed(1)}%</div>
                        </div>
                    </div>
                </div>

                {/* Requests Table */}
                <div className="glass card" style={{ padding: '0' }}>
                    <div style={{ padding: '24px 32px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <h3 style={{ fontSize: "16px", fontWeight: 700 }}>Request Logs</h3>
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', color: 'var(--muted)' }}>
                                    <th style={{ padding: '16px 32px', fontWeight: 600 }}>TIMESTAMP</th>
                                    <th style={{ padding: '16px 32px', fontWeight: 600 }}>MODEL</th>
                                    <th style={{ padding: '16px 32px', fontWeight: 600 }}>OPTIMIZED</th>
                                    <th style={{ padding: '16px 32px', fontWeight: 600 }}>LATENCY</th>
                                    <th style={{ padding: '16px 32px', fontWeight: 600, textAlign: 'right' }}>ACTION</th>
                                </tr>
                            </thead>
                            <tbody>
                                {recentLogs.map((req) => (
                                    <tr key={req.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                                        <td style={{ padding: '16px 32px', opacity: 0.6 }}>{new Date(req.timestamp * 1000).toLocaleString()}</td>
                                        <td style={{ padding: '16px 32px', fontWeight: 600 }}>{req.model.split('/').pop()}</td>
                                        <td style={{ padding: '16px 32px' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent)' }}>
                                                <Zap size={12} />
                                                {((req.tokens_saved / (req.tokens_in + req.tokens_saved || 1)) * 100).toFixed(1)}%
                                            </div>
                                        </td>
                                        <td style={{ padding: '16px 32px', opacity: 0.6 }}>{req.latency_ms.toFixed(0)}ms</td>
                                        <td style={{ padding: '16px 32px', textAlign: 'right' }}>
                                            <button
                                                onClick={() => router.push(`/analytics/${req.id}`)}
                                                className="btn glass"
                                                style={{ padding: '8px 16px', fontSize: '11px', fontWeight: 700, borderRadius: '8px' }}
                                            >
                                                INSPECT <ChevronRight size={12} style={{ marginLeft: '4px' }} />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {!loading && recentLogs.length === 0 && (
                            <div style={{ textAlign: 'center', padding: '64px', opacity: 0.2 }}>
                                <Search size={48} style={{ margin: '0 auto 16px' }} />
                                <p>No logs available.</p>
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
