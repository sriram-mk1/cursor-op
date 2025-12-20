"use client";

import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { motion } from "framer-motion";
import { ArrowLeft, Clock, Database, Zap, Shield, Cpu, Code, MessageSquare, ChevronRight, Activity, Layers } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { createClient } from "@/utils/supabase";

export default function RequestDetailPage() {
    const { id } = useParams();
    const [log, setLog] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
    const supabase = createClient();

    const renderContent = (content: any) => {
        if (typeof content === 'string') return content;
        if (Array.isArray(content)) {
            return content.map((c, i) => (
                <span key={i}>{typeof c === 'string' ? c : (c.text || JSON.stringify(c))}</span>
            ));
        }
        if (typeof content === 'object' && content !== null) {
            return content.text || JSON.stringify(content);
        }
        return String(content || "");
    };

    useEffect(() => {
        const fetchDetail = async () => {
            // Since we don't have a direct endpoint yet, we fetch from Supabase
            const { data, error } = await supabase
                .from("analytics")
                .select("*")
                .eq("id", id)
                .single();

            if (data) setLog(data);
            setLoading(false);
        };
        fetchDetail();
    }, [id, supabase]);

    if (loading) return <div className="layout-wrapper"><Sidebar /><main className="main-content"><div style={{ opacity: 0.3, padding: "40px" }}>Loading detail...</div></main></div>;
    if (!log) return <div className="layout-wrapper"><Sidebar /><main className="main-content"><div style={{ padding: "40px" }}>Request not found.</div></main></div>;

    return (
        <div className="layout-wrapper">
            <Sidebar />

            <main className="main-content">
                <header style={{ marginBottom: "32px" }}>
                    <Link href="/" style={{ display: "inline-flex", alignItems: "center", gap: "8px", fontSize: "12px", color: "var(--accent)", marginBottom: "16px", textDecoration: "none" }}>
                        <ArrowLeft size={14} /> Back to Overview
                    </Link>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                        <div>
                            <h1 style={{ fontSize: "20px", fontWeight: 600, display: "flex", alignItems: "center", gap: "10px" }}>
                                Deep Observability
                                <span style={{ fontSize: "10px", background: "rgba(255,255,255,0.05)", padding: "4px 8px", borderRadius: "4px", color: "var(--muted)" }}>{log.id}</span>
                            </h1>
                            <p className="text-muted" style={{ fontSize: "13px", marginTop: "4px" }}>Surgical inspection of request flow and context reconstruction.</p>
                        </div>
                        <div className="glass" style={{ padding: "8px 16px", fontSize: "12px", fontWeight: 600, background: "rgba(0,255,136,0.05)", border: "1px solid var(--accent)" }}>
                            <Zap size={12} className="text-accent" style={{ verticalAlign: "middle", marginRight: "8px" }} />
                            SAVED {log.tokens_saved.toLocaleString()} TOKENS
                        </div>
                    </div>
                </header>

                <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1.8fr", gap: "32px" }}>
                    {/* Metadata Column */}
                    <section className="glass card" style={{ padding: "24px", height: "fit-content" }}>
                        <h2 style={{ fontSize: "12px", fontWeight: 700, textTransform: "uppercase", marginBottom: "20px", opacity: 0.4 }}>Request Metadata</h2>
                        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
                            <MetaItem icon={<Cpu size={14} />} label="Model" value={log.model} />
                            <MetaItem icon={<Clock size={14} />} label="Timestamp" value={new Date(log.timestamp * 1000).toLocaleString()} />
                            <MetaItem icon={<Code size={14} />} label="Session ID" value={log.session_id} mono />
                            <MetaItem icon={<Activity size={14} />} label="Latency" value={`${log.latency_ms.toFixed(0)}ms`} />
                            <MetaItem icon={<Database size={14} />} label="Provider" value={log.or_id ? "OpenRouter" : "Internal"} />

                            <div style={{ paddingTop: "24px", borderTop: "1px solid var(--border)", marginTop: "8px" }}>
                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                                    <div>
                                        <div style={{ fontSize: "10px", opacity: 0.4, marginBottom: "4px" }}>RAW COST</div>
                                        <div style={{ fontSize: "14px", fontWeight: 600 }}>${(log.total_cost_usd || 0).toFixed(6)}</div>
                                    </div>
                                    <div>
                                        <div style={{ fontSize: "10px", color: "var(--accent)", marginBottom: "4px" }}>SAVED COST</div>
                                        <div style={{ fontSize: "14px", fontWeight: 600, color: "var(--accent)" }}>${(log.cost_saved_usd || 0).toFixed(6)}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Interaction Column */}
                    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
                        {/* Raw Interaction */}
                        <section className="glass card" style={{ padding: "0" }}>
                            <div style={{ padding: "20px 24px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <h2 style={{ fontSize: "13px", fontWeight: 600, display: "flex", alignItems: "center", gap: "8px" }}>
                                    <MessageSquare size={16} className="text-muted" />
                                    Conversation Flow
                                </h2>
                            </div>
                            <div style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "24px", maxHeight: "600px", overflowY: "auto" }}>
                                {log.raw_messages?.map((msg: any, i: number) => (
                                    <div key={i} style={{ borderLeft: `2px solid ${msg.role === 'user' ? 'var(--accent)' : 'rgba(255,255,255,0.1)'}`, paddingLeft: "16px" }}>
                                        <div style={{ fontSize: "10px", fontWeight: 700, textTransform: "uppercase", marginBottom: "8px", opacity: 0.4 }}>
                                            {msg.role}
                                        </div>
                                        <div style={{ fontSize: "13px", lineHeight: "1.6", whiteSpace: "pre-wrap", color: "rgba(255,255,255,0.8)" }}>
                                            {renderContent(msg.content)}
                                        </div>
                                    </div>
                                ))}
                                {log.response_message && (
                                    <div style={{ borderLeft: `2px solid var(--purple)`, paddingLeft: "16px" }}>
                                        <div style={{ fontSize: "10px", fontWeight: 700, textTransform: "uppercase", marginBottom: "8px", color: "var(--purple)" }}>
                                            ASSISTANT (RESPONSE)
                                        </div>
                                        <div style={{ fontSize: "13px", lineHeight: "1.6", whiteSpace: "pre-wrap", color: "rgba(255,255,255,0.9)" }}>
                                            {renderContent(log.response_message.content)}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </section>

                        {/* Reconstruction Sequence (If available) */}
                        {log.reconstruction_log?.sequence && (
                            <section className="glass card" style={{ padding: "24px" }}>
                                <h2 style={{ fontSize: "13px", fontWeight: 600, marginBottom: "20px", display: "flex", alignItems: "center", gap: "8px" }}>
                                    <Layers size={16} className="text-accent" />
                                    Reconstruction Sequence
                                </h2>
                                <div style={{ display: "flex", flexDirection: "column", gap: "2px", background: "rgba(255,255,255,0.02)", borderRadius: "8px", overflow: "hidden" }}>
                                    {log.reconstruction_log.sequence.slice(0, 20).map((item: any, i: number) => (
                                        <div key={i} style={{ display: "grid", gridTemplateColumns: "40px 100px 1fr 60px", padding: "8px 16px", fontSize: "11px", alignItems: "center", background: item.score > 0 ? "rgba(0,255,136,0.03)" : "transparent" }}>
                                            <span style={{ opacity: 0.2, fontFamily: "monospace" }}>{item.line_index.toString().padStart(3, '0')}</span>
                                            <span style={{ opacity: 0.4, fontWeight: 700 }}>{item.source.toUpperCase()}</span>
                                            <span style={{ opacity: 0.8, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{item.text}</span>
                                            <span style={{ textAlign: "right", color: item.score > 0 ? "var(--accent)" : "rgba(255,255,255,0.1)" }}>
                                                {item.score > 0 ? (item.score * 100).toFixed(0) + "%" : "---"}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </section>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}

function MetaItem({ icon, label, value, mono }: any) {
    return (
        <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "11px", color: "var(--muted)", marginBottom: "6px" }}>
                {icon}
                {label}
            </div>
            <div style={{ fontSize: "13px", fontWeight: 500, fontFamily: mono ? "monospace" : "inherit", color: "var(--fg)" }}>
                {value}
            </div>
        </div>
    );
}
