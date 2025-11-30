import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export const ThirdChart = () => (
  <div className="col-span-2 bg-gray-50 rounded-[16px] p-5 flex flex-col gap-4">
    <div className="text-[20px] font-semibold">Использование по ОС</div>
    <div style={{ width: "100%", height: 120 }}>
      <ResponsiveContainer>
        <BarChart
          data={[
            { name: "Linux", value: 8000 },
            { name: "Mac", value: 22000 },
            { name: "iOS", value: 18000 },
            { name: "Windows", value: 24000 },
            { name: "Android", value: 15000 },
            { name: "Other", value: 20000 },
          ]}
        >
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  </div>
);
