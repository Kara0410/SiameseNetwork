import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { DeploymentGroup } from "@/lib/types";

interface DeploymentMapProps {
  groups: DeploymentGroup[];
}

export function DeploymentMap({ groups }: DeploymentMapProps) {
  return (
    <Card>
      <CardHeader>
        <div>
          <CardDescription>Deployment</CardDescription>
          <CardTitle>Production fullstack</CardTitle>
        </div>
        <Badge variant="success">open source</Badge>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3.5 sm:grid-cols-2">
          {groups.map((group) => (
            <div key={group.category} className="rounded-lg border border-border bg-white/[0.035] p-4">
              <span className="text-xs uppercase tracking-[0.14em] text-muted">{group.category}</span>
              <div className="mt-3 flex flex-wrap gap-2">
                {group.items.map((item) => (
                  <span
                    key={item}
                    className="rounded-pill border border-border bg-white/[0.07] px-3 py-1.5 text-sm text-foreground"
                  >
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
