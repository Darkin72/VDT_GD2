import {
  AbsoluteFill,
  CalculateMetadataFunction,
  Composition,
  Easing,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import type { CSSProperties } from "react";

type Props = {
  title: string;
  subtitle: string;
  accent: string;
};

const calculateMetadata: CalculateMetadataFunction<Props> = () => ({
  durationInFrames: 180,
  fps: 30,
  width: 1280,
  height: 720,
});

const clouds = [
  { x: 90, y: 120, scale: 1.1, speed: 0.35 },
  { x: 610, y: 170, scale: 0.75, speed: 0.2 },
  { x: 1040, y: 105, scale: 0.9, speed: 0.28 },
];

export const MyComposition = () => {
  return (
    <Composition
      id="MyComp"
      component={MyComponent}
      durationInFrames={180}
      fps={30}
      width={1280}
      height={720}
      defaultProps={{
        title: "VỊT ĐẠP XE",
        subtitle: "Một chuyến đi thật vui",
        accent: "#f7c948",
      }}
      calculateMetadata={calculateMetadata}
    />
  );
};

const Cloud: React.FC<{ x: number; y: number; scale: number }> = ({ x, y, scale }) => (
  <g transform={`translate(${x} ${y}) scale(${scale})`} fill="#fff" opacity={0.82}>
    <ellipse cx="0" cy="18" rx="62" ry="22" />
    <circle cx="-30" cy="4" r="25" />
    <circle cx="4" cy="-9" r="34" />
    <circle cx="38" cy="8" r="24" />
  </g>
);

export const MyComponent: React.FC<Props> = ({ title, subtitle, accent }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const entrance = spring({ frame, fps, config: { damping: 16, stiffness: 100 } });
  const exit = interpolate(frame, [154, 180], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  const ride = interpolate(frame, [0, 180], [-390, 390], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.quad),
  });
  const bounce = Math.sin(frame * 0.48) * 3;
  const pedal = frame * 13;
  const legSwing = Math.sin(frame * 0.48) * 19;
  const dashOffset = (frame * 10) % 160;

  const shell: CSSProperties = {
    backgroundColor: "#b9ebff",
    color: "#17324d",
    overflow: "hidden",
    fontFamily: "Trebuchet MS, Arial, sans-serif",
  };

  return (
    <AbsoluteFill style={shell}>
      <svg width="100%" height="100%" viewBox="0 0 1280 720" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#8edcff" />
            <stop offset="100%" stopColor="#e6f9ff" />
          </linearGradient>
          <linearGradient id="road" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#4b6372" />
            <stop offset="100%" stopColor="#263541" />
          </linearGradient>
          <filter id="shadow" x="-30%" y="-30%" width="160%" height="180%">
            <feGaussianBlur stdDeviation="8" />
          </filter>
        </defs>

        <rect width="1280" height="720" fill="url(#sky)" />
        <circle cx="1080" cy="115" r="58" fill="#fff4b0" opacity="0.95" />
        {clouds.map((cloud, index) => (
          <Cloud
            key={index}
            x={cloud.x - ((frame * cloud.speed) % 1450)}
            y={cloud.y}
            scale={cloud.scale}
          />
        ))}

        <path d="M0 415 Q180 365 360 410 T720 400 T1080 405 T1440 390 L1440 500 L0 500Z" fill="#8dcf83" />
        <path d="M0 455 Q220 410 430 448 T850 438 T1280 430 L1280 520 L0 520Z" fill="#5fae70" />
        <path d="M0 502 L1280 502 L1280 720 L0 720Z" fill="url(#road)" />
        <path d="M0 515 L1280 515" stroke="#d3e4d7" strokeWidth="5" opacity="0.55" />
        {Array.from({ length: 11 }, (_, index) => (
          <rect key={index} x={index * 160 - dashOffset} y="610" width="82" height="10" rx="5" fill="#f9e7a3" opacity="0.9" />
        ))}

        <ellipse cx={640 + ride} cy="568" rx="190" ry="18" fill="#13222b" opacity="0.28" filter="url(#shadow)" />

        <g transform={`translate(${640 + ride} ${bounce})`} opacity={entrance * exit}>
          <g transform="translate(0 2)">
            <circle cx="-130" cy="500" r="70" fill="none" stroke="#1e3442" strokeWidth="10" />
            <circle cx="130" cy="500" r="70" fill="none" stroke="#1e3442" strokeWidth="10" />
            <g stroke="#98c5d1" strokeWidth="3" opacity="0.9">
              {Array.from({ length: 8 }, (_, index) => (
                <g key={index} transform={`rotate(${index * 45} -130 500)`}>
                  <line x1="-130" y1="500" x2="-130" y2="438" />
                </g>
              ))}
              {Array.from({ length: 8 }, (_, index) => (
                <g key={index} transform={`rotate(${index * 45} 130 500)`}>
                  <line x1="130" y1="500" x2="130" y2="438" />
                </g>
              ))}
            </g>
            <circle cx="-130" cy="500" r="9" fill="#f7c948" />
            <circle cx="130" cy="500" r="9" fill="#f7c948" />
            <path d="M-130 500 L-38 432 L40 500 L130 500 M-38 432 L0 370" fill="none" stroke="#f7c948" strokeWidth="10" strokeLinejoin="round" />
            <path d="M-45 430 L-20 405" stroke="#1e3442" strokeWidth="8" strokeLinecap="round" />
            <path d="M-20 405 L18 405" stroke="#1e3442" strokeWidth="8" strokeLinecap="round" />
            <g transform={`rotate(${pedal} 40 500)`}>
              <line x1="40" y1="500" x2="68" y2="500" stroke="#1e3442" strokeWidth="6" />
              <circle cx="74" cy="500" r="8" fill="#f7c948" />
            </g>
            <path d="M0 370 L-10 330" stroke="#1e3442" strokeWidth="8" strokeLinecap="round" />
            <path d="M-10 330 L-2 300" stroke="#1e3442" strokeWidth="8" strokeLinecap="round" />
          </g>

          <g transform={`translate(0 ${Math.sin(frame * 0.48) * 2})`}>
            <ellipse cx="0" cy="330" rx="72" ry="86" fill="#f7c948" stroke="#d9972b" strokeWidth="5" />
            <path d="M-62 316 Q-98 335 -79 363 Q-57 379 -37 350" fill="#eeb93c" stroke="#d9972b" strokeWidth="5" />
            <ellipse cx="-40" cy="365" rx="20" ry="12" fill="#eeb93c" transform={`rotate(${legSwing} -40 365)`} />
            <ellipse cx="42" cy="365" rx="20" ry="12" fill="#eeb93c" transform={`rotate(${-legSwing} 42 365)`} />
            <circle cx="0" cy="235" r="58" fill="#f7c948" stroke="#d9972b" strokeWidth="5" />
            <path d="M-48 224 Q0 180 50 224 L42 244 Q0 218 -42 244Z" fill="#ef4f58" />
            <path d="M-50 224 Q0 184 50 224" fill="none" stroke="#d13a45" strokeWidth="8" />
            <circle cx="21" cy="227" r="8" fill="#17324d" />
            <circle cx="24" cy="224" r="3" fill="#fff" />
            <path d="M48 245 L100 260 L48 278 Q36 260 48 245Z" fill="#f29f32" stroke="#d27a22" strokeWidth="4" />
            <path d="M-55 300 Q-102 305 -105 265" fill="none" stroke="#d9972b" strokeWidth="16" strokeLinecap="round" />
            <path d="M-105 265 Q-89 248 -70 260" fill="none" stroke="#d9972b" strokeWidth="11" strokeLinecap="round" />
          </g>
        </g>
      </svg>

      <div style={{ position: "absolute", top: 36, left: 0, right: 0, textAlign: "center", opacity: entrance * exit, transform: `translateY(${interpolate(entrance, [0, 1], [-24, 0])}px)` }}>
        <div style={{ color: "#17324d", fontSize: 17, fontWeight: 700, letterSpacing: "0.22em" }}>DUCK ON TOUR</div>
        <div style={{ marginTop: 10, color: accent, fontSize: 52, fontWeight: 900, letterSpacing: "-0.04em", textShadow: "0 3px 0 #fff" }}>{title}</div>
        <div style={{ marginTop: 4, color: "#31566d", fontSize: 20 }}>{subtitle}</div>
      </div>

      <div style={{ position: "absolute", left: 48, right: 48, bottom: 28, display: "flex", alignItems: "center", gap: 14, color: "#dceef0", fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace", fontSize: 12, letterSpacing: "0.12em", opacity: exit }}>
        <span>WHEELS: {String(Math.round((frame / 180) * 100)).padStart(3, "0")}%</span>
        <div style={{ height: 4, flex: 1, borderRadius: 2, background: "#314855" }}>
          <div style={{ height: "100%", width: `${(frame / 180) * 100}%`, borderRadius: 2, background: accent }} />
        </div>
        <span>QUACK!</span>
      </div>
    </AbsoluteFill>
  );
};
