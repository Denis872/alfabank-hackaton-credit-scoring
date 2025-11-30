import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { monthlyData } from "./mocks";

export const FirstChart = () => (
  <div className="col-span-1 bg-gray-50 p-5 flex flex-col gap-4 rounded-[12px]">
    <div className="text-[20px] font-semibold">
      Доходы в году / Траты в году
    </div>
    <div style={{ width: "100%", height: 250 }}>
      <ResponsiveContainer>
        <LineChart
          data={monthlyData}
          margin={{ top: 20, right: 20, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="month" />
          <YAxis />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="2025"
            stroke="#ff7a7a"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="2024"
            stroke="#7ea6ff"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  </div>
);
