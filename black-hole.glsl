precision highp float;
uniform vec2 resolution;
uniform float time;
uniform vec2 mouse;

#define MAX_STEPS 512

#define SCHWARZSCHILD_RADIUS 0.4
#define PI 3.14159265359

vec3 diskColor(float dist)
{
    vec3 hotColor = vec3(1.0, 1.0, 1.0);  // hot
    vec3 midColor = vec3(0.5, 0.5, 0.5);  // middle
    vec3 coolColor = vec3(0.2, 0.2, 0.2); // cooler

    float innerTemp = smoothstep(2.0, 4.0, dist);
    float outerTemp = smoothstep(4.0, 8.0, dist);

    vec3 color = mix(hotColor, midColor, innerTemp);
    color = mix(color, coolColor, outerTemp);

    // add temperature-based intensity
    float intensity = exp(-dist * 0.25);
    return color * intensity;
}

float rand(vec2 co)
{
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 rayMarch(vec3 ro, vec3 rd, float radius)
{
    vec3 p = ro;
    float r = length(p);
    vec3 col = vec3(0.0);
    float totalDist = 0.0;
    float stepSize = 0.1;

    for (int i = 0; i < MAX_STEPS; i++)
    {
        r = length(p);

        float force = radius / (r * r);
        vec3 gravity = normalize(p) * -force;

        // update ray direction based on gravity
        rd += gravity * stepSize;
        rd = normalize(rd);

        // move along the ray
        p += rd * stepSize;
        totalDist += stepSize;

        // check if we're in the accretion disk region
        float diskDist = abs(p.y);
        float diskRadius = length(vec2(p.x, p.z));

        if (diskDist < 0.1 && diskRadius > 2.0 * radius)
        {
            // calculate disk color based on temperature/distance
            vec3 diskCol = diskColor(diskRadius) * rand(p.xy + time * 0.000001);

            // add time-based rotation
            float angle = atan(p.z, p.x) + time * 0.2;
            diskCol *= 1.0 + 0.2 * sin(angle * 8.0);

            // accumulate color with distance falloff
            col += diskCol * exp(-totalDist * 0.07);
        }
    }

    return col;
}

void main()
{
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / resolution.y;
    vec2 aspect = resolution.xy / resolution.y;
    vec2 muv = (mouse.xy - 0.5) * aspect;
    float pixelate = smoothstep(0., .05, sqrt(dot(uv - muv, uv - muv)));

    // setup camera
    vec3 ro = vec3(0.0, 2.0, -20.0) + pixelate;
    vec3 rd = normalize(vec3(uv, 1.0));

    // ray march the scene
    vec3 col = rayMarch(ro, rd, SCHWARZSCHILD_RADIUS);

    gl_FragColor = vec4(col, 1.0);
}