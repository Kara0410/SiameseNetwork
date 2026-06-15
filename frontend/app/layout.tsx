import type { Metadata } from "next";
import { Inter, IBM_Plex_Mono } from "next/font/google";

import { Topbar } from "@/components/layout/topbar";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

// Geist is unavailable via next/font/google on Next 14; Inter is the closest
// production-grade match for the design system's "Geist" typeface.
const geist = Inter({ subsets: ["latin"], variable: "--font-geist" });
const ibmPlexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-ibm-plex-mono" });

export const metadata: Metadata = {
  title: "VisionIQ — Identity Intelligence Platform",
  description: "Multimodal AI identity verification, explainability, and vector search.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`dark ${geist.variable} ${ibmPlexMono.variable}`}>
      <body className="min-h-screen font-sans text-foreground antialiased">
        <TooltipProvider>
          <div className="px-4 py-4 sm:px-6">
            <Topbar />
            <div className="mx-auto max-w-6xl pt-6">{children}</div>
          </div>
        </TooltipProvider>
      </body>
    </html>
  );
}
