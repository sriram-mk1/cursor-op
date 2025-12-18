"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { User, Shield, Database } from "lucide-react";
import { createClient } from "@/utils/supabase";

export default function SettingsPage() {
    const [user, setUser] = useState<any>(null);
    const supabase = createClient();

    useEffect(() => {
        const getUser = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            setUser(user);
        };
        getUser();
    }, [supabase.auth]);

    return (
        <div className="layout-wrapper">
            <Sidebar />

            <main className="main-content">
                <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "48px" }}>
                    <div>
                        <h1 style={{ fontSize: "24px", marginBottom: "4px" }}>Settings</h1>
                        <p className="text-muted" style={{ fontSize: "14px" }}>Configure your V1 Gateway preferences and account settings.</p>
                    </div>
                </header>

                <div style={{ display: "grid", gap: "32px", maxWidth: "800px" }}>
                    <section>
                        <h3 style={{ fontSize: "16px", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
                            <User size={18} /> Profile Settings
                        </h3>
                        <div className="glass card" style={{ display: "grid", gap: "20px" }}>
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                                <div>
                                    <label className="stat-label">Full Name</label>
                                    <input type="text" className="input" placeholder="Not set" style={{ width: "100%", marginTop: "8px" }} />
                                </div>
                                <div>
                                    <label className="stat-label">Email Address</label>
                                    <input type="email" className="input" value={user?.email || ""} style={{ width: "100%", marginTop: "8px" }} disabled />
                                </div>
                            </div>
                            <button className="btn" style={{ width: "fit-content" }}>Save Changes</button>
                        </div>
                    </section>

                    <section>
                        <h3 style={{ fontSize: "16px", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
                            <Database size={18} /> Gateway Configuration
                        </h3>
                        <div className="glass card" style={{ display: "grid", gap: "20px" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <div>
                                    <div style={{ fontSize: "14px", fontWeight: 500 }}>Context Optimization</div>
                                    <div className="text-muted" style={{ fontSize: "12px" }}>Automatically prune and optimize context for all requests.</div>
                                </div>
                                <div style={{ width: "40px", height: "20px", background: "var(--accent)", borderRadius: "10px", position: "relative" }}>
                                    <div style={{ width: "16px", height: "16px", background: "black", borderRadius: "50%", position: "absolute", right: "2px", top: "2px" }} />
                                </div>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <div>
                                    <div style={{ fontSize: "14px", fontWeight: 500 }}>Streaming Optimization</div>
                                    <div className="text-muted" style={{ fontSize: "12px" }}>Enable real-time context pruning for streaming responses.</div>
                                </div>
                                <div style={{ width: "40px", height: "20px", background: "var(--border)", borderRadius: "10px", position: "relative" }}>
                                    <div style={{ width: "16px", height: "16px", background: "white", borderRadius: "50%", position: "absolute", left: "2px", top: "2px" }} />
                                </div>
                            </div>
                        </div>
                    </section>

                    <section>
                        <h3 style={{ fontSize: "16px", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
                            <Shield size={18} /> Danger Zone
                        </h3>
                        <div className="glass card" style={{ borderColor: "rgba(255, 68, 68, 0.2)" }}>
                            <p className="text-muted" style={{ fontSize: "13px", marginBottom: "16px" }}>
                                Deleting your account will permanently remove all API keys, analytics, and session data. This action cannot be undone.
                            </p>
                            <button className="btn btn-outline" style={{ color: "#ff4444", borderColor: "rgba(255, 68, 68, 0.2)" }}>Delete Account</button>
                        </div>
                    </section>
                </div>
            </main>
        </div>
    );
}
