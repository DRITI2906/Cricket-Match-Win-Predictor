import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, ArrowUp, ArrowDown } from "lucide-react";

interface FeatureImpact {
  feature: string;
  impact: number;
  direction: "positive" | "negative";
}

interface ShapExplanationProps {
  features: FeatureImpact[];
}

export const ShapExplanation = ({ features }: ShapExplanationProps) => {
  return (
    <Card className="shadow-[var(--shadow-card)]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-primary" />
          Explainable AI - SHAP Analysis
        </CardTitle>
        <CardDescription>
          Understanding which factors influenced the prediction most
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {features.map((feature, index) => {
            const absImpact = Math.abs(feature.impact);
            const percentage = (absImpact * 100).toFixed(1);
            const isPositive = feature.direction === "positive";

            return (
              <div key={index} className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    {isPositive ? (
                      <ArrowUp className="w-4 h-4 text-primary" />
                    ) : (
                      <ArrowDown className="w-4 h-4 text-destructive" />
                    )}
                    <span className="font-medium">{feature.feature}</span>
                  </div>
                  <span className={`font-semibold ${isPositive ? 'text-primary' : 'text-destructive'}`}>
                    {isPositive ? '+' : ''}{percentage}%
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      isPositive ? 'bg-primary' : 'bg-destructive'
                    }`}
                    style={{ width: `${absImpact * 100}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-6 p-4 bg-muted/50 rounded-lg space-y-2">
          <div className="text-sm font-semibold">How to read this:</div>
          <ul className="text-xs text-muted-foreground space-y-1 list-disc list-inside">
            <li>Green bars show features increasing win probability</li>
            <li>Red bars show features decreasing win probability</li>
            <li>Larger bars indicate stronger influence on the prediction</li>
            <li>SHAP values explain individual prediction transparency</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};
