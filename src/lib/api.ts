// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// API types matching backend schema
export interface PredictionRequest {
  team1: string;
  team2: string;
  venue: string;
  toss_winner?: string;
  toss_decision?: string;
  match_type?: string;
  runs_required?: number;
  balls_remaining?: number;
  wickets_in_hand?: number;
  target_match?: number;
  current_run_rate?: number;
  required_run_rate?: number;
}

export interface ShapValue {
  feature: string;
  value: number;
  impact: "positive" | "negative" | "neutral";
}

export interface PredictionResponse {
  winner: string;
  probability: number;
  confidence: "high" | "medium" | "low";
  shap_explanation: ShapValue[];
  factors: {
    toss: string;
    toss_decision?: string;
    venue: string;
    match_type: string;
  };
}

// API service
export class CricketPredictionAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<{ status: string }> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) {
      throw new Error('API health check failed');
    }
    return response.json();
  }

  /**
   * Predict match outcome
   */
  async predict(data: PredictionRequest): Promise<PredictionResponse> {
    const response = await fetch(`${this.baseUrl}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `API request failed with status ${response.status}`);
    }

    return response.json();
  }
}

// Export singleton instance
export const api = new CricketPredictionAPI();
