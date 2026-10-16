import {
  Interactive,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Stage, Pipeline, clamp, colors } from "./Visuals";

const MetricCard = ({
  left,
  label,
  value,
  description,
  delay,
  accent = false,
}: {
  left: number;
  label: string;
  value: string;
  description: string;
  delay: number;
  accent?: boolean;
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <div
      style={{
        position: "absolute",
        left,
        top: 350,
        width: 330,
        height: 190,
        boxSizing: "border-box",
        padding: "24px 26px",
        background: "#FFFFFFC7",
        borderTop: `7px solid ${accent ? colors.red : colors.ink}`,
        opacity: interpolate(frame, [delay * fps, (delay + 0.7) * fps], [0, 1], clamp),
        translate: interpolate(frame, [delay * fps, (delay + 0.7) * fps], ["0px 18px", "0px 0px"], clamp),
      }}
    >
      <div style={{ fontSize: 21, color: colors.muted, fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 47, color: accent ? colors.red : colors.ink, fontWeight: 700, marginTop: 13 }}>
        {value}
      </div>
      <div style={{ fontSize: 21, color: colors.muted, marginTop: 12 }}>{description}</div>
    </div>
  );
};

const DenResultInner = () => (
  <Stage
    chapter="07 / KẾT QUẢ DEN"
    title="Kết quả mô hình dùng DEN"
    subtitle="Mô hình gốc được đánh giá trên bộ dữ liệu SOTS-OTS."
  >
    <div style={{ position: "absolute", left: 70, top: 225 }}>
      <Pipeline active={4} refinement="DEN" />
    </div>
    <MetricCard left={70} label="CHẤT LƯỢNG" value="32,35 dB" description="PSNR trên SOTS-OTS" delay={0.8} accent />
    <MetricCard left={475} label="TỐC ĐỘ" value="28,04 FPS" description="Tốc độ suy luận" delay={1.4} />
    <MetricCard left={880} label="THỜI GIAN" value="35,66 ms" description="Thời gian mỗi ảnh" delay={2} />
    <div
      style={{
        position: "absolute",
        left: 72,
        top: 595,
        right: 70,
        fontSize: 25,
        fontWeight: 700,
        color: colors.ink,
      }}
    >
      Các giá trị này là mốc so sánh khi thay DEN bằng DSRB.
    </div>
  </Stage>
);

export const DenResult = Interactive.withSchema({
  Component: DenResultInner,
  componentName: "HazeClearDenResult",
  schema: {},
  wrapInSequence: true,
  layout: "absolute-fill",
});
