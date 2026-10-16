import {
  Interactive,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Stage, clamp, colors } from "./Visuals";

const Block = ({
  children,
  left,
  top,
  width,
  accent = false,
  delay,
}: {
  children: React.ReactNode;
  left: number;
  top: number;
  width: number;
  accent?: boolean;
  delay: number;
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        minHeight: 84,
        boxSizing: "border-box",
        padding: "19px 15px",
        border: `3px solid ${accent ? colors.red : colors.ink}`,
        borderRadius: 16,
        background: accent ? "#FBE4E8" : "#E7F0F3",
        textAlign: "center",
        fontSize: 22,
        lineHeight: 1.18,
        fontWeight: 700,
        opacity: interpolate(frame, [delay * fps, (delay + 0.7) * fps], [0, 1], clamp),
        scale: interpolate(frame, [delay * fps, (delay + 0.7) * fps], [0.85, 1], clamp),
      }}
    >
      {children}
    </div>
  );
};

const Arrow = ({ left, top, width, delay }: { left: number; top: number; width: number; delay: number }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        color: colors.ink,
        fontSize: 42,
        textAlign: "center",
        opacity: interpolate(frame, [delay * fps, (delay + 0.4) * fps], [0, 1], clamp),
      }}
    >
      →
    </div>
  );
};

const OptimizationInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const resultOpacity = interpolate(frame, [5.2 * fps, 6.2 * fps], [0, 1], clamp);
  return (
    <Stage
      chapter="06 / DSRB THAY DEN"
      title="Giảm chi phí bằng tích chập tách theo chiều sâu"
      subtitle="Giữ Wavelet và LFRN; chỉ thay khối tinh chỉnh cuối bằng DSRB."
      note="So sánh DSRB với baseline trên cùng SOTS-OTS • Không so trực tiếp với profiling RESIDE-6K"
    >
      <div style={{ position: "absolute", left: 65, top: 222, fontSize: 25, color: colors.red, fontWeight: 700 }}>
        CƠ CHẾ DSRB
      </div>
      <Block left={65} top={280} width={145} delay={0.7}>Đặc trưng<br />từ IWT</Block>
      <Arrow left={211} top={300} width={55} delay={1.3} />
      <Block left={265} top={270} width={175} accent delay={1.6}>Depthwise 3×3<br />xử lý từng kênh</Block>
      <Arrow left={441} top={300} width={55} delay={2.2} />
      <Block left={495} top={270} width={180} delay={2.5}>Pointwise 1×1<br />trộn thông tin kênh</Block>
      <div
        style={{
          position: "absolute",
          left: 114,
          top: 381,
          width: 470,
          height: 95,
          borderLeft: `3px solid ${colors.red}`,
          borderBottom: `3px solid ${colors.red}`,
          opacity: interpolate(frame, [3.1 * fps, 3.8 * fps], [0, 1], clamp),
        }}
      />
      <Block left={495} top={440} width={180} accent delay={3.7}>+ Residual</Block>
      <div style={{ position: "absolute", left: 67, top: 540, width: 610, fontSize: 22, color: colors.muted }}>
        Residual giữ đường truyền đặc trưng; depthwise giảm chi phí không gian và pointwise trộn thông tin kênh.
      </div>

      <div
        style={{
          position: "absolute",
          left: 740,
          top: 222,
          width: 475,
          opacity: resultOpacity,
        }}
      >
        <div style={{ fontSize: 25, color: colors.red, fontWeight: 700 }}>KẾT QUẢ SOTS-OTS</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, marginTop: 30 }}>
          <div style={{ background: "#FFFFFFC8", padding: 20, borderTop: `6px solid ${colors.ink}` }}>
            <div style={{ fontSize: 20, color: colors.muted }}>BASELINE</div>
            <div style={{ fontSize: 43, fontWeight: 700, marginTop: 8 }}>28,04 FPS</div>
            <div style={{ fontSize: 24, marginTop: 9 }}>32,35 dB</div>
          </div>
          <div style={{ background: "#FFFFFFC8", padding: 20, borderTop: `6px solid ${colors.red}` }}>
            <div style={{ fontSize: 20, color: colors.red }}>DSRB</div>
            <div style={{ fontSize: 43, fontWeight: 700, color: colors.red, marginTop: 8 }}>37,26 FPS</div>
            <div style={{ fontSize: 24, marginTop: 9 }}>31,18 dB</div>
          </div>
        </div>
        <div style={{ marginTop: 26, fontSize: 31, lineHeight: 1.35, fontWeight: 700 }}>
          <span style={{ color: colors.red }}>+32,87% FPS</span>
          <br />
          <span style={{ color: colors.teal }}>−24,73% thời gian</span>
        </div>
        <div style={{ marginTop: 22, fontSize: 23, lineHeight: 1.35 }}>
          PSNR giảm <b>1,17 dB</b>: tốc độ đạt mục tiêu, chất lượng chưa đạt giới hạn giảm tối đa 0,3 dB.
        </div>
      </div>
    </Stage>
  );
};

export const Optimization = Interactive.withSchema({
  Component: OptimizationInner,
  componentName: "HazeClearOptimization",
  schema: {},
  wrapInSequence: true,
  layout: "absolute-fill",
});
