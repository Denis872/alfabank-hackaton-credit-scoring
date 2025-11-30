import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";
import { COLORS, pieData } from "./mocks";

export const SecondChart = () => (
  <div className="col-span-1 bg-gray-50 p-5 rounded-[12px]">
    <div className="text-[20px] font-semibold">Портфель покупок</div>
    <div style={{ width: "100%", height: 200 }} className="mt-2">
      <ResponsiveContainer>
        <PieChart>
          <Pie
            data={pieData}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={70}
          >
            {pieData.map((_, index) => (
              <Cell
                fill={COLORS[index % COLORS.length]}
                key={`cell-${index}`}
              />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  </div>
);
