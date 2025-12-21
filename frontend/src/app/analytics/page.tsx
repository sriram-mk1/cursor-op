"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { TrendingUp, Zap, ChevronRight, Search, ChevronLeft } from "lucide-react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { createClient } from "@/utils/supabase";
import { AreaChart, Area, ResponsiveContainer, Tooltip, XAxis } from "recharts";

export default function AnalyticsPage() {
    const router = useRouter();
    const [stats, setStats] = useState<any>(null);
    const [requests, setRequests] = useState<any[]>([]);
    const [view, setView] = useState("daily");
    const [loading, setLoading] = useState(true);

    // Pagination
    const [page, setPage] = useState(1);
    const ITEMS_PER_PAGE = 7;

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
                    // Mock data for dev if empty
                    if (fetchedReqs.length === 0 && process.env.NODE_ENV === 'development') {
                        const mockData = Array.from({ length: 20 }).map((_, i) => ({
                            id: `mock-${i}`,
                            model: i % 2 === 0 ? 'anthropic/claude-3-5-sonnet' : 'openai/gpt-4o',
                            tokens_in: 5000 + Math.random() * 5000,
                            tokens_saved: 2000 + Math.random() * 3000,
                            latency_ms: 500 + Math.random() * 1000,
                            cost_saved_usd: 0.002 + Math.random() * 0.005,
                            timestamp: (Date.now() / 1000) - (i * 3600)
                        }));
                        setRequests(mockData);
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

    // Aggregates
    const totalSaved = recentLogs.reduce((acc: number, r: any) => acc + (r.cost_saved_usd || 0), 0);
    const totalTokensSaved = recentLogs.reduce((acc: number, r: any) => acc + (r.tokens_saved || 0), 0);
    const avgLatency = recentLogs.length > 0 ? recentLogs.reduce((acc: number, r: any) => acc + r.latency_ms, 0) / recentLogs.length : 0;
    const avgOptimization = recentLogs.length > 0 ? (recentLogs.reduce((acc: number, r: any) => acc + r.tokens_saved, 0) / recentLogs.reduce((acc: number, r: any) => acc + (r.tokens_in + r.tokens_saved), 0)) * 100 : 0;

    // Chart Data Preparation
    const chartData = [...recentLogs].reverse().map(r => ({
        time: new Date(r.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        savings: r.cost_saved_usd || 0,
        tokens: r.tokens_saved || 0
    }));

    // Pagination Logic
    const totalPages = Math.ceil(recentLogs.length / ITEMS_PER_PAGE);
    const paginatedLogs = recentLogs.slice((page - 1) * ITEMS_PER_PAGE, page * ITEMS_PER_PAGE);

    return (
        <div className="layout-wrapper">
            <Sidebar />

            <main className="main-content">
                <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "48px" }}>
                    <div>
                        <h1 style={{ fontSize: "24px", fontWeight: 700, letterSpacing: '-0.03em' }}>Analytics Dashboard</h1>
                        <p className="text-muted" style={{ fontSize: "14px", marginTop: "4px" }}>Performance monitoring and context optimization metrics.</p>
                    </div>
                    {/* View Toggle - Squared, Simple */}
                    <div className="glass" style={{ padding: "0", borderRadius: "0", display: "flex", gap: "1px", border: '1px solid rgba(255,255,255,0.1)' }}>
                        {["daily", "weekly", "monthly"].map(v => (
                            <button
                                key={v}
                                onClick={() => setView(v)}
                                className="btn-simple"
                                style={{
                                    background: view === v ? "var(--fg)" : "transparent",
                                    color: view === v ? "var(--bg)" : "var(--fg)",
                                    fontSize: "10px",
                                    padding: "8px 16px",
                                    fontWeight: 700,
                                    textTransform: "uppercase",
                                    borderRadius: 0,
                                    border: 'none',
                                    cursor: 'pointer'
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
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "24px" }}>
                            <div>
                                <h3 style={{ fontSize: "11px", fontWeight: 800, color: "var(--muted)", textTransform: 'uppercase', letterSpacing: '0.05em' }}>SAVINGS TREND</h3>
                                <div style={{ fontSize: "28px", fontWeight: 600, marginTop: "8px" }}>${totalSaved.toFixed(4)}</div>
                            </div>
                            <div className="savings-badge">
                                <TrendingUp size={12} /> {avgOptimization.toFixed(1)}% Optimized
                            </div>
                        </div>

                        <div style={{ height: '220px', marginLeft: '-20px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={chartData}>
                                    <defs>
                                        <linearGradient id="colorSaved" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="var(--accent)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <XAxis
                                        dataKey="time"
                                        stroke="rgba(255,255,255,0.2)"
                                        fontSize={10}
                                        tickLine={false}
                                        axisLine={false}
                                        interval="preserveStartEnd"
                                    />
                                    <Tooltip
                                        contentStyle={{ background: '#0a0a0a', border: '1px solid rgba(255,255,255,0.1)', fontSize: '12px' }}
                                        labelStyle={{ color: 'rgba(255,255,255,0.5)', marginBottom: '4px' }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="tokens"
                                        stroke="var(--accent)"
                                        fillOpacity={1}
                                        fill="url(#colorSaved)"
                                        strokeWidth={2}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Secondary Stats */}
                    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                        <div className="glass card stat-item" style={{ background: "rgba(var(--accent-rgb), 0.05)", borderColor: "var(--accent)" }}>
                            <div className="stat-label" style={{ color: "var(--accent)" }}>Net Cost Savings</div>
                            <div className="stat-value" style={{ color: "var(--accent)" }}>${(totalSaved).toFixed(4)}</div>
                            <div style={{ fontSize: "11px", opacity: 0.6 }}>Cumulative across all sessions</div>
                        </div>
                        <div className="glass card stat-item">
                            <div className="stat-label">Total Tokens Saved</div>
                            <div className="stat-value">{totalTokensSaved.toLocaleString()}</div>
                        </div>
                        <div className="glass card stat-item">
                            <div className="stat-label">Avg. Latency</div>
                            <div className="stat-value">{avgLatency.toFixed(0)} <span style={{ fontSize: "14px", opacity: 0.4 }}>ms</span></div>
                        </div>
                        <div className="glass card stat-item">
                            <div className="stat-label">Reduction Rate</div>
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
                                {paginatedLogs.map((req) => (
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
                                                className="btn-inspect"
                                                style={{
                                                    padding: '8px 12px',
                                                    fontSize: '10px',
                                                    fontWeight: 800,
                                                    borderRadius: '0',
                                                    background: 'transparent',
                                                    border: '1px solid rgba(255,255,255,0.1)',
                                                    color: 'var(--fg)',
                                                    cursor: 'pointer',
                                                    minWidth: '60px',
                                                    letterSpacing: '0.05em'
                                                }}
                                                onMouseEnter={(e) => {
                                                    e.currentTarget.style.borderColor = 'var(--fg)';
                                                    e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                                                }}
                                                onMouseLeave={(e) => {
                                                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)';
                                                    e.currentTarget.style.background = 'transparent';
                                                }}
                                            >
                                                INSPECT
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

                        {/* Pagination Controls */}
                        {recentLogs.length > ITEMS_PER_PAGE && (
                            <div style={{ padding: '16px 32px', display: 'flex', justifyContent: 'flex-end', gap: '8px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                                <button
                                    onClick={() => setPage(p => Math.max(1, p - 1))}
                                    disabled={page === 1}
                                    style={{
                                        padding: '8px 12px', background: 'none', border: '1px solid rgba(255,255,255,0.1)',
                                        color: page === 1 ? 'rgba(255,255,255,0.2)' : 'var(--fg)', cursor: page === 1 ? 'default' : 'pointer'
                                    }}
                                >
                                    <ChevronLeft size={14} />
                                </button>
                                <span style={{ display: 'flex', alignItems: 'center', fontSize: '12px', opacity: 0.5, padding: '0 8px' }}>
                                    PAGE {page} OF {totalPages}
                                </span>
                                <button
                                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                                    disabled={page === totalPages}
                                    style={{
                                        padding: '8px 12px', background: 'none', border: '1px solid rgba(255,255,255,0.1)',
                                        color: page === totalPages ? 'rgba(255,255,255,0.2)' : 'var(--fg)', cursor: page === totalPages ? 'default' : 'pointer'
                                    }}
                                >
                                    <ChevronRight size={14} />
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
