"use client";

import { useState, useEffect } from "react";
import { LayoutDashboard, Key, BarChart3, Settings, LogOut, Zap, PanelLeftClose, PanelLeft } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createClient } from "@/utils/supabase";

export function Sidebar() {
    const pathname = usePathname();
    const router = useRouter();
    const [isCollapsed, setIsCollapsed] = useState(true); // Default to collapsed
    const [user, setUser] = useState<any>(null);
    const supabase = createClient();

    useEffect(() => {
        const saved = localStorage.getItem("sidebar_collapsed");
        if (saved !== null) {
            setIsCollapsed(saved === "true");
        }

        const getUser = async () => {
            const { data: { user } } = await supabase.auth.getUser();
            setUser(user);
        };
        getUser();
    }, [supabase.auth]);

    const toggleCollapse = (e: React.MouseEvent) => {
        e.stopPropagation();
        const newState = !isCollapsed;
        setIsCollapsed(newState);
        localStorage.setItem("sidebar_collapsed", newState.toString());
    };

    const handleSignOut = async (e: React.MouseEvent) => {
        e.stopPropagation();
        await supabase.auth.signOut();
        router.push("/login");
    };

    const navItems = [
        { icon: LayoutDashboard, label: "Overview", href: "/" },
        { icon: Key, label: "API Keys", href: "/keys" },
        { icon: BarChart3, label: "Deep Observability", href: "/observability" },
        { icon: Settings, label: "Settings", href: "/settings" },
    ];

    return (
        <aside
            className={`sidebar ${isCollapsed ? "collapsed" : ""}`}
            onClick={() => isCollapsed && setIsCollapsed(false)}
            style={{ cursor: isCollapsed ? "pointer" : "default" }}
        >
            <div style={{ display: "flex", alignItems: "center", justifyContent: isCollapsed ? "center" : "space-between", padding: "0 4px", minHeight: "32px", marginBottom: "12px" }}>
                {!isCollapsed && (
                    <div style={{ display: "flex", alignItems: "center", gap: "12px", paddingLeft: "4px" }}>
                        <div style={{ background: "var(--accent)", padding: "4px", borderRadius: "4px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                            <Zap size={14} color="black" />
                        </div>
                        <span style={{ fontWeight: 700, fontSize: "16px", letterSpacing: "-0.03em" }}>V1</span>
                    </div>
                )}
                {isCollapsed && (
                    <div style={{ background: "var(--accent)", padding: "6px", borderRadius: "4px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <Zap size={16} color="black" />
                    </div>
                )}
                {!isCollapsed && (
                    <button
                        onClick={toggleCollapse}
                        style={{
                            background: "none",
                            border: "none",
                            color: "rgba(255,255,255,0.3)",
                            cursor: "pointer",
                            padding: "4px",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            transition: "color 0.2s"
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.color = "var(--fg)"}
                        onMouseLeave={(e) => e.currentTarget.style.color = "rgba(255,255,255,0.3)"}
                    >
                        <PanelLeftClose size={16} />
                    </button>
                )}
            </div>

            <nav style={{ flex: 1, display: "flex", flexDirection: "column", gap: "4px" }}>
                {navItems.map((item) => {
                    const isActive = pathname === item.href;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            onClick={(e) => {
                                // Prevent the aside's onClick from firing and expanding the sidebar
                                e.stopPropagation();
                            }}
                            style={{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: isCollapsed ? "center" : "flex-start",
                                gap: "12px",
                                padding: "10px 12px",
                                borderRadius: "var(--radius)",
                                fontSize: "13px",
                                color: isActive ? "var(--accent)" : "rgba(255,255,255,0.6)",
                                background: isActive ? "var(--accent-muted)" : "transparent",
                                transition: "all 0.2s ease",
                                whiteSpace: "nowrap"
                            }}
                            className="nav-link"
                        >
                            <item.icon size={18} />
                            {!isCollapsed && <span>{item.label}</span>}
                        </Link>
                    );
                })}
            </nav>

            <div style={{ display: "flex", flexDirection: "column", gap: "12px", padding: "8px 0" }}>
                {!isCollapsed && user && (
                    <div style={{ padding: "0 12px", marginBottom: "4px" }}>
                        <div style={{ fontSize: "11px", color: "rgba(255,255,255,0.4)", marginBottom: "4px" }}>SIGNED IN AS</div>
                        <div style={{ fontSize: "12px", fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis" }}>{user.email}</div>
                    </div>
                )}
                <button
                    onClick={handleSignOut}
                    className="btn btn-outline"
                    style={{
                        width: "100%",
                        justifyContent: isCollapsed ? "center" : "flex-start",
                        border: "none",
                        padding: "12px 16px",
                        height: "auto",
                        background: "rgba(255,255,255,0.05)",
                        borderRadius: "var(--radius)",
                        transition: "all 0.2s ease"
                    }}
                >
                    <LogOut size={18} />
                    {!isCollapsed && <span style={{ fontWeight: 600 }}>Sign Out</span>}
                </button>
            </div>
        </aside>
    );
}
