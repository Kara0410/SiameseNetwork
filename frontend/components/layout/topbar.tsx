"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard" },
  { href: "/architecture", label: "Architecture" },
];

export function Topbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-4 z-20 mx-auto flex max-w-6xl items-center justify-between gap-4 rounded-pill border border-border bg-background-raised/70 px-3 py-2.5 backdrop-blur-2xl">
      <div className="flex items-center gap-2.5 pl-1 text-sm font-extrabold tracking-tight text-foreground">
        <span className="h-7 w-7 rounded-full bg-[conic-gradient(from_80deg,#68E8FF,#8BFFCA,#B9A5FF,#68E8FF)] shadow-[0_0_28px_rgba(104,232,255,0.42)]" />
        VisionIQ
      </div>
      <nav className="flex gap-1 rounded-pill bg-white/[0.04] p-1">
        {NAV_ITEMS.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "rounded-pill px-4 py-2 text-sm font-medium text-muted transition-colors hover:text-foreground",
                active && "bg-accent/[0.18] text-foreground"
              )}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="hidden items-center gap-2 text-xs text-muted sm:flex">
        <span className="h-2 w-2 rounded-full bg-success shadow-[0_0_14px_#8BFFCA]" />
        model v2.6 online
      </div>
    </header>
  );
}
