--
-- PostgreSQL database dump
--

-- Dumped from database version 16.4
-- Dumped by pg_dump version 16.4

-- Started on 2024-11-24 15:14:02

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 218 (class 1259 OID 16449)
-- Name: consultas; Type: TABLE; Schema: public; Owner: usuario_uva
--

CREATE TABLE public.consultas (
    id integer NOT NULL,
    user_id integer,
    class_name character varying(100),
    consulta_fecha timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.consultas OWNER TO usuario_uva;

--
-- TOC entry 217 (class 1259 OID 16448)
-- Name: consultas_id_seq; Type: SEQUENCE; Schema: public; Owner: usuario_uva
--

CREATE SEQUENCE public.consultas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.consultas_id_seq OWNER TO usuario_uva;

--
-- TOC entry 4872 (class 0 OID 0)
-- Dependencies: 217
-- Name: consultas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: usuario_uva
--

ALTER SEQUENCE public.consultas_id_seq OWNED BY public.consultas.id;


--
-- TOC entry 215 (class 1259 OID 16412)
-- Name: usuarios; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.usuarios (
    id integer NOT NULL,
    nombre character varying(255) NOT NULL,
    correo character varying(255) NOT NULL,
    "contraseña" character varying(255) NOT NULL,
    fecha_registro timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    estado boolean DEFAULT true,
    rol character varying(10) DEFAULT 'user'::character varying NOT NULL,
    CONSTRAINT chk_rol CHECK (((rol)::text = ANY ((ARRAY['user'::character varying, 'admin'::character varying])::text[])))
);


ALTER TABLE public.usuarios OWNER TO postgres;

--
-- TOC entry 216 (class 1259 OID 16419)
-- Name: usuarios_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.usuarios_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.usuarios_id_seq OWNER TO postgres;

--
-- TOC entry 4874 (class 0 OID 0)
-- Dependencies: 216
-- Name: usuarios_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.usuarios_id_seq OWNED BY public.usuarios.id;


--
-- TOC entry 220 (class 1259 OID 16578)
-- Name: validaciones; Type: TABLE; Schema: public; Owner: usuario_uva
--

CREATE TABLE public.validaciones (
    id integer NOT NULL,
    user_id integer NOT NULL,
    nombre_usuario character varying(255) NOT NULL,
    frame_path character varying(500) NOT NULL,
    user_class character varying(50) NOT NULL,
    model_class character varying(50) NOT NULL,
    fecha_validacion timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.validaciones OWNER TO usuario_uva;

--
-- TOC entry 219 (class 1259 OID 16577)
-- Name: validaciones_id_seq; Type: SEQUENCE; Schema: public; Owner: usuario_uva
--

CREATE SEQUENCE public.validaciones_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.validaciones_id_seq OWNER TO usuario_uva;

--
-- TOC entry 4876 (class 0 OID 0)
-- Dependencies: 219
-- Name: validaciones_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: usuario_uva
--

ALTER SEQUENCE public.validaciones_id_seq OWNED BY public.validaciones.id;


--
-- TOC entry 4702 (class 2604 OID 16452)
-- Name: consultas id; Type: DEFAULT; Schema: public; Owner: usuario_uva
--

ALTER TABLE ONLY public.consultas ALTER COLUMN id SET DEFAULT nextval('public.consultas_id_seq'::regclass);


--
-- TOC entry 4698 (class 2604 OID 16422)
-- Name: usuarios id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.usuarios ALTER COLUMN id SET DEFAULT nextval('public.usuarios_id_seq'::regclass);


--
-- TOC entry 4704 (class 2604 OID 16581)
-- Name: validaciones id; Type: DEFAULT; Schema: public; Owner: usuario_uva
--

ALTER TABLE ONLY public.validaciones ALTER COLUMN id SET DEFAULT nextval('public.validaciones_id_seq'::regclass);


--
-- TOC entry 4863 (class 0 OID 16449)
-- Dependencies: 218
-- Data for Name: consultas; Type: TABLE DATA; Schema: public; Owner: usuario_uva
--

COPY public.consultas (id, user_id, class_name, consulta_fecha) FROM stdin;
1	12	Mildiú	2024-10-27 17:13:04.247275
2	12	Mildiú	2024-10-27 17:14:44.249432
3	12	Tizón de la hoja	2024-10-27 17:18:01.38063
4	12	Tizón de la hoja	2024-10-27 17:34:49.241813
5	12	Tizón de la hoja	2024-10-27 17:36:29.694109
6	12	Mildiú	2024-10-27 17:37:03.172802
7	12	Mildiú	2024-10-27 17:37:37.925643
8	12	Botrytis cinerea	2024-10-27 17:47:39.597345
9	12	Tizón de la hoja	2024-10-27 18:12:46.470898
10	12	Botrytis cinerea	2024-10-27 18:50:24.402709
11	12	Mildiú	2024-10-28 01:53:45.043853
12	12	Tizón de la hoja	2024-10-28 02:06:57.816174
13	12	Mildiú	2024-10-28 02:25:57.582588
14	12	Tizón de la hoja	2024-10-28 12:31:57.184751
15	12	Mildiú	2024-10-28 12:32:18.084166
16	12	Tizón de la hoja	2024-10-28 12:39:33.866151
17	12	Mildiú	2024-10-28 12:39:50.075007
18	12	Mildiú	2024-10-28 12:40:03.231912
19	12	Esca	2024-10-28 12:40:03.26685
20	12	Mildiú	2024-10-28 12:48:27.928361
21	12	Mildiú	2024-10-28 12:48:52.678854
22	12	Saludable	2024-10-28 12:49:09.593901
23	12	Mildiú	2024-10-28 12:49:20.43255
24	12	Tizón de la hoja	2024-10-28 12:56:06.106124
25	12	Mildiú	2024-10-28 12:56:33.92046
26	12	Esca, Mildiú	2024-10-28 12:57:18.453561
27	12	Mildiú	2024-10-28 13:08:20.612063
28	12	Tizón de la hoja	2024-10-29 03:21:27.679233
29	12	Mildiú	2024-10-29 03:27:48.739768
30	12	Tizón de la hoja	2024-10-29 03:40:07.404364
31	12	Mildiú	2024-10-29 03:53:19.550611
32	12	Mildiú	2024-10-29 03:55:15.409721
33	12	Mildiú	2024-10-29 04:04:13.895708
34	12	Tizón de la hoja	2024-10-29 04:52:19.800984
35	12	Mildiú	2024-10-29 04:56:04.386275
36	12	Botrytis cinerea	2024-10-29 10:50:11.985229
37	12	Mildiú	2024-10-29 10:52:37.029456
38	12	Mildiú	2024-10-29 11:10:58.081115
39	12	Tizón de la hoja	2024-10-29 11:16:15.08208
40	12	Mildiú	2024-10-29 11:17:23.546133
41	12	Mildiú	2024-10-29 11:25:04.243305
42	12	Mildiú	2024-10-29 11:28:17.300298
43	12	Botrytis cinerea	2024-10-29 11:35:21.134232
44	12	Mildiú	2024-10-29 11:42:41.145431
45	12	Mildiú	2024-10-29 11:47:09.610367
46	12	Mildiú	2024-10-29 11:48:36.287306
47	12	Tizón de la hoja	2024-10-29 15:29:07.777479
48	12	Esca, Mildiú	2024-10-29 15:30:16.789396
49	12	Esca, Mildiú	2024-10-29 15:35:05.076011
50	12	Mildiú	2024-10-29 15:38:14.579756
51	12	Mildiú	2024-10-29 15:41:46.949251
52	12	Mildiú	2024-10-29 15:47:17.07266
53	12	Esca, Mildiú	2024-10-29 15:47:42.440112
54	12	Mildiú	2024-10-29 15:51:19.733979
55	12	Esca, Mildiú	2024-10-29 15:52:10.881433
56	12	Esca, Mildiú	2024-10-29 15:57:17.782581
57	12	Botrytis cinerea	2024-10-29 15:58:22.782106
58	12	Mildiú	2024-10-29 15:58:57.998705
59	12	Tizón de la hoja	2024-10-29 18:28:02.396788
60	12	Mildiú, Esca	2024-10-29 18:48:00.22769
61	12	Tizón de la hoja	2024-10-31 00:09:21.895481
62	12	Esca, Mildiú	2024-10-31 00:09:43.376212
63	12	Tizón de la hoja	2024-10-31 00:46:32.834387
64	12	Tizón de la hoja	2024-10-31 00:57:56.08055
65	12	Oídio, Mildiú	2024-10-31 00:59:36.719838
66	12	Clase temporal	2024-10-31 02:07:20.590872
67	12	Mildiú	2024-10-31 02:08:59.065702
68	12	Esca	2024-10-31 02:09:25.385546
69	12	Oídio	2024-10-31 02:09:34.377894
70	12	Oídio	2024-10-31 02:09:40.142448
71	12	Oídio	2024-10-31 02:09:47.726785
72	12	Podredumbre negra	2024-10-31 02:09:56.505804
73	12	Mildiú	2024-10-31 02:10:08.150228
74	12	Saludable	2024-10-31 02:10:14.252525
75	12	Saludable	2024-10-31 02:10:20.234239
76	12	Saludable	2024-10-31 02:10:25.110268
77	12	Mildiú	2024-10-31 02:10:52.358122
78	12	Botrytis cinerea	2024-10-31 02:11:19.146971
79	12	Saludable	2024-10-31 02:12:26.725497
80	12	Mildiú	2024-10-31 02:12:50.114698
81	12	Mildiú	2024-10-31 02:13:15.04839
82	12	Mildiú	2024-10-31 02:14:11.980901
83	12	Mildiú	2024-10-31 02:14:16.397555
84	12	Mildiú	2024-10-31 02:14:19.235383
85	12	Oídio	2024-10-31 02:14:40.708509
86	12	Botrytis cinerea, Mildiú	2024-10-31 02:23:16.032006
87	12	Botrytis cinerea, Mildiú	2024-10-31 02:24:18.166793
88	12	Mildiú	2024-10-31 02:25:00.154833
89	12	Saludable	2024-10-31 02:25:43.996912
90	12	Botrytis cinerea	2024-10-31 02:25:53.997209
91	12	Mildiú	2024-10-31 02:36:23.916331
92	12	Saludable	2024-10-31 02:36:31.992937
93	12	Mildiú, Oídio, Saludable	2024-10-31 02:37:08.952463
94	12	Botrytis cinerea, Mildiú, Oídio	2024-10-31 02:37:09.171164
95	12	Mildiú	2024-10-31 02:45:59.870847
96	12	Mildiú, Oídio, Botrytis cinerea	2024-10-31 02:47:30.742101
97	12	Mildiú, Botrytis cinerea	2024-10-31 02:48:37.592446
98	12	Tizón de la hoja	2024-10-31 02:56:23.0519
99	12	Tizón de la hoja	2024-10-31 02:56:28.069848
100	12	Tizón de la hoja	2024-10-31 02:56:34.98575
101	12	Tizón de la hoja	2024-10-31 02:56:40.57271
102	12	Podredumbre negra	2024-10-31 02:56:47.995885
103	12	Oídio	2024-10-31 02:57:04.467225
104	12	Mildiú	2024-10-31 02:57:14.754438
105	12	Botrytis cinerea	2024-10-31 02:57:28.78473
106	12	Botrytis cinerea	2024-10-31 02:57:40.348505
107	12	Podredumbre negra	2024-10-31 02:57:53.216015
108	12	Mildiú, Saludable, Oídio	2024-10-31 03:06:44.850251
109	12	Oídio, Mildiú, Saludable	2024-10-31 09:37:50.837373
110	12	Mildiú	2024-11-04 11:14:45.775784
111	12	Botrytis cinerea	2024-11-04 11:14:57.075499
112	12	Mildiú	2024-11-04 11:15:01.590582
113	12	Mildiú	2024-11-04 11:15:05.516746
114	12	Botrytis cinerea	2024-11-04 11:15:14.806397
115	12	Mildiú, Botrytis cinerea	2024-11-04 12:46:18.62819
116	12	Mildiú	2024-11-04 12:47:36.971148
117	12	Oídio	2024-11-04 13:04:59.466573
118	12	Mildiú, Oídio, Saludable	2024-11-04 13:16:41.20914
119	12	Mildiú	2024-11-04 13:59:32.374929
120	12	Mildiú	2024-11-04 13:59:47.351566
121	12	Oídio	2024-11-05 02:31:36.749349
122	12	Mildiú, Botrytis cinerea	2024-11-05 02:31:54.84663
123	12	Oídio	2024-11-05 02:41:15.807891
124	12	Mildiú	2024-11-05 02:48:12.061674
125	12	Mildiú	2024-11-05 03:02:10.216121
126	12	Oídio	2024-11-05 03:12:56.277431
127	12	Botrytis cinerea, Mildiú	2024-11-05 03:13:21.141833
128	12	Mildiú	2024-11-05 03:15:07.069992
129	12	Oídio	2024-11-05 03:34:41.511959
130	12	Mildiú	2024-11-05 03:34:55.092341
131	12	Oídio	2024-11-05 03:34:55.144334
132	12	Mildiú	2024-11-05 03:34:55.225065
133	12	Mildiú	2024-11-05 03:34:55.280437
134	12	Mildiú	2024-11-05 03:34:55.331605
135	12	Mildiú	2024-11-05 03:41:30.818523
136	12	Mildiú	2024-11-05 03:59:29.496723
137	12	Mildiú	2024-11-05 03:59:33.47999
138	12	Mildiú	2024-11-05 03:59:37.397152
139	12	Oídio	2024-11-05 04:08:14.825072
140	12	Oídio	2024-11-05 04:08:22.997095
141	12	Saludable	2024-11-05 04:08:26.904573
142	12	Oídio	2024-11-05 04:20:48.07486
143	12	Botrytis cinerea, Mildiú	2024-11-05 04:21:30.987414
144	12	Oídio	2024-11-05 04:25:40.515407
145	12	Mildiú, Botrytis cinerea	2024-11-05 04:26:09.216998
146	12	Mildiú	2024-11-05 04:29:11.833209
147	12	Mildiú	2024-11-05 10:43:50.995638
148	12	Mildiú	2024-11-05 10:43:58.914214
149	12	Oídio	2024-11-05 10:43:58.989957
150	12	Mildiú	2024-11-05 10:43:59.099801
151	12	Mildiú	2024-11-05 10:43:59.174392
152	12	Mildiú	2024-11-05 10:43:59.244226
153	12	Saludable	2024-11-05 10:43:59.373638
154	12	Botrytis cinerea, Mildiú	2024-11-05 10:44:13.879283
155	12	Mildiú	2024-11-05 10:50:41.855955
156	12	Mildiú, Oídio, Saludable	2024-11-05 10:50:55.872127
157	12	Mildiú	2024-11-05 10:53:58.81021
158	12	Mildiú	2024-11-05 10:54:27.878115
159	12	Oídio	2024-11-05 10:58:36.916566
160	12	Mildiú	2024-11-05 10:58:43.837658
161	12	Mildiú	2024-11-05 10:58:52.146795
162	12	Oídio	2024-11-05 10:58:52.22664
163	12	Mildiú	2024-11-05 10:58:52.348975
164	12	Mildiú	2024-11-05 10:58:52.434807
165	12	Mildiú	2024-11-05 10:58:52.515247
166	12	Saludable	2024-11-05 10:58:52.647948
167	12	Botrytis cinerea, Mildiú	2024-11-05 10:59:28.975021
168	12	Botrytis cinerea, Mildiú	2024-11-05 11:03:04.207278
169	12	\N	2024-11-05 11:03:05.656418
170	12	Mildiú	2024-11-05 11:10:24.368018
171	12	Mildiú	2024-11-05 11:10:41.69665
172	12	Mildiú, Botrytis cinerea	2024-11-05 11:10:55.988285
173	12	Oídio	2024-11-05 11:25:16.310922
174	12	Botrytis cinerea	2024-11-05 11:25:38.767665
175	12	Mildiú	2024-11-05 11:37:11.394069
176	12	Botrytis cinerea, Mildiú	2024-11-05 11:37:35.979274
177	12	Botrytis cinerea, Mildiú	2024-11-05 11:37:38.835778
178	12	Oídio	2024-11-05 13:12:10.90809
179	12	Botrytis cinerea, Mildiú	2024-11-05 13:13:31.299787
180	12	Botrytis cinerea, Mildiú	2024-11-05 13:22:58.021576
181	12	Mildiú	2024-11-05 13:29:18.429494
182	12	Oídio, Mildiú, Saludable	2024-11-05 13:30:09.449646
183	12	Oídio	2024-11-05 13:30:40.505979
184	12	Mildiú, Botrytis cinerea	2024-11-05 13:31:02.011166
185	12	Oídio	2024-11-05 13:33:13.844701
186	12	Oídio	2024-11-05 13:39:30.594063
187	12	Mildiú	2024-11-05 14:55:39.058231
188	12	Mildiú, Botrytis cinerea	2024-11-05 14:56:26.633669
189	12	Oídio, Saludable, Mildiú	2024-11-05 17:31:56.505122
190	12	Botrytis cinerea, Mildiú	2024-11-05 17:33:19.574469
191	12	Botrytis cinerea, Mildiú	2024-11-05 17:34:36.537448
192	12	Oídio, Saludable, Mildiú	2024-11-05 17:34:52.687735
193	12	Oídio	2024-11-08 03:49:35.886596
194	12	Mildiú	2024-11-08 04:13:05.002022
195	12	Oídio	2024-11-08 04:58:25.153029
196	12	Mildiú	2024-11-08 04:59:47.702147
197	12	Mildiú	2024-11-08 05:08:14.574076
198	12	Mildiú	2024-11-08 05:32:42.85883
199	12	Oídio	2024-11-08 05:39:18.364956
200	12	Oídio	2024-11-08 05:43:25.183535
201	12	Oídio	2024-11-08 05:45:29.70079
202	12	Mildiú	2024-11-08 06:01:25.426254
203	12	Oídio	2024-11-08 06:10:50.369861
204	12	Mildiú	2024-11-08 06:14:45.263857
205	12	Oídio	2024-11-08 06:18:34.78612
206	12	Oídio	2024-11-08 06:32:33.730426
207	12	Oídio	2024-11-08 06:33:45.380012
208	12	Oídio	2024-11-08 06:36:00.853737
209	12	Oídio	2024-11-08 06:38:52.26659
210	12	Oídio	2024-11-08 13:51:04.769988
211	12	Oídio	2024-11-08 14:00:32.72588
212	12	Mildiú	2024-11-08 16:52:27.461287
213	12	Mildiú, Botrytis cinerea	2024-11-08 16:58:36.504964
214	12	Oídio, Mildiú, Saludable	2024-11-08 17:13:46.520786
215	12	Oídio	2024-11-08 17:43:55.708329
216	12	Botrytis cinerea	2024-11-10 13:26:37.835405
217	12	Mildiú	2024-11-10 13:31:55.134173
218	12	Mildiú	2024-11-10 13:41:23.822888
219	12	Mildiú	2024-11-10 13:46:44.233764
220	12	Mildiú	2024-11-10 17:06:38.07119
221	12	Mildiú	2024-11-10 17:14:09.812756
222	12	Mildiú	2024-11-10 17:18:22.412643
223	12	Mildiú	2024-11-10 17:25:33.842173
224	12	Saludable	2024-11-10 17:32:58.359674
225	12	Mildiú	2024-11-10 17:36:04.414294
226	12	Mildiú	2024-11-10 17:49:17.709226
227	12	Mildiú	2024-11-10 18:17:51.477938
228	12	Mildiú	2024-11-10 18:18:54.006534
229	12	Mildiú, Botrytis cinerea	2024-11-10 18:26:57.675324
230	12	Mildiú	2024-11-11 11:18:19.277769
231	12	Mildiú	2024-11-11 11:35:24.203442
232	12	Mildiú, Botrytis cinerea	2024-11-11 11:35:36.580858
233	16	Mildiú	2024-11-11 11:59:23.777239
234	16	Mildiú	2024-11-11 11:59:28.346067
235	16	Oídio	2024-11-11 11:59:28.424387
236	16	Mildiú	2024-11-11 11:59:28.553763
237	16	Mildiú	2024-11-11 11:59:28.638505
238	16	Mildiú	2024-11-11 11:59:28.715812
239	16	Saludable	2024-11-11 11:59:28.849972
240	16	Botrytis cinerea, Mildiú	2024-11-11 11:59:46.170909
241	12	Mildiú	2024-11-11 13:14:07.780603
242	12	Mildiú	2024-11-11 13:41:27.52596
243	12	Botrytis cinerea, Mildiú	2024-11-11 13:41:37.884845
244	12	Botrytis cinerea, Mildiú	2024-11-11 13:47:17.031864
245	12	Mildiú	2024-11-11 14:01:31.56781
246	12	Mildiú, Botrytis cinerea	2024-11-11 14:01:39.590088
247	12	Botrytis cinerea, Mildiú	2024-11-11 14:09:42.656877
248	12	Botrytis cinerea, Mildiú	2024-11-11 14:12:36.638533
249	12	Mildiú	2024-11-11 23:17:34.041638
250	12	Mildiú, Botrytis cinerea	2024-11-11 23:17:48.758732
251	12	Mildiú, Botrytis cinerea	2024-11-11 23:48:10.772237
252	12	Mildiú, Botrytis cinerea	2024-11-12 01:10:56.129161
253	12	Mildiú, Botrytis cinerea	2024-11-12 01:11:39.894261
254	12	Botrytis cinerea, Mildiú	2024-11-12 08:09:15.966997
255	12	Mildiú	2024-11-12 08:15:05.652617
256	12	Botrytis cinerea, Mildiú	2024-11-12 08:15:14.465132
257	12	Botrytis cinerea, Mildiú	2024-11-12 08:16:02.727622
258	12	Mildiú, Botrytis cinerea	2024-11-12 08:36:50.341342
259	12	Mildiú, Botrytis cinerea	2024-11-12 08:37:18.316739
260	12	Mildiú, Botrytis cinerea	2024-11-12 08:45:35.302161
261	12	Mildiú, Botrytis cinerea	2024-11-12 08:47:04.327292
262	12	Mildiú, Botrytis cinerea	2024-11-12 09:20:03.634811
263	12	Mildiú, Botrytis cinerea	2024-11-12 09:23:21.627352
264	12	Botrytis cinerea, Mildiú	2024-11-12 09:26:33.149984
265	12	Mildiú, Botrytis cinerea	2024-11-12 09:33:18.649179
266	12	Botrytis cinerea, Mildiú	2024-11-12 09:36:58.606053
267	12	Mildiú, Botrytis cinerea	2024-11-12 10:33:43.384931
268	12	Botrytis cinerea, Mildiú	2024-11-12 10:44:56.265142
269	12	Botrytis cinerea, Mildiú	2024-11-12 10:53:59.190097
270	12	Mildiú, Botrytis cinerea	2024-11-12 11:02:09.916424
271	12	Mildiú, Botrytis cinerea	2024-11-12 11:08:14.918062
272	12	Mildiú, Botrytis cinerea	2024-11-12 11:25:22.604386
273	12	Mildiú, Botrytis cinerea	2024-11-12 11:39:19.204258
274	12	Botrytis cinerea, Mildiú	2024-11-12 11:56:24.035396
275	12	Mildiú, Botrytis cinerea	2024-11-12 12:06:03.388394
276	12	Botrytis cinerea, Mildiú	2024-11-12 12:11:06.55409
277	12	Botrytis cinerea, Mildiú	2024-11-12 12:14:29.670123
278	12	Botrytis cinerea, Mildiú	2024-11-12 12:22:10.162636
279	12	Mildiú, Botrytis cinerea	2024-11-12 12:26:04.380791
280	12	Botrytis cinerea, Mildiú	2024-11-12 12:30:45.394036
281	12	Botrytis cinerea, Mildiú	2024-11-12 12:34:45.256648
282	12	Oídio, Mildiú, Saludable	2024-11-12 12:40:20.026498
283	12	Botrytis cinerea, Mildiú	2024-11-12 12:42:49.884479
284	12	Botrytis cinerea, Mildiú	2024-11-12 12:54:59.526887
285	12	Botrytis cinerea, Mildiú	2024-11-12 12:59:27.343295
286	12	Botrytis cinerea, Mildiú	2024-11-12 13:03:31.505908
287	12	Mildiú, Botrytis cinerea	2024-11-12 13:07:04.017755
288	12	Mildiú, Botrytis cinerea	2024-11-12 13:11:51.505381
289	12	Mildiú, Botrytis cinerea	2024-11-12 13:14:09.784625
290	12	Mildiú, Botrytis cinerea	2024-11-12 13:20:16.462494
291	12	Botrytis cinerea, Mildiú	2024-11-12 14:05:22.189537
292	12	Botrytis cinerea, Mildiú	2024-11-12 14:12:03.634519
293	12	Botrytis cinerea, Mildiú	2024-11-12 14:17:37.020704
294	12	Mildiú, Botrytis cinerea	2024-11-12 14:22:37.02776
295	12	Mildiú, Botrytis cinerea	2024-11-12 14:28:32.539154
296	12	Oídio, Saludable, Mildiú	2024-11-12 14:38:44.211621
297	12	Botrytis cinerea, Mildiú	2024-11-12 14:44:26.103924
298	12	Botrytis cinerea, Mildiú	2024-11-12 14:48:54.560513
299	12	Botrytis cinerea, Mildiú	2024-11-12 14:51:14.755752
300	12	Mildiú, Botrytis cinerea	2024-11-12 14:55:38.345772
301	12	Mildiú, Botrytis cinerea	2024-11-12 14:59:47.526225
302	12	Botrytis cinerea, Mildiú	2024-11-12 15:02:32.882653
303	12	Mildiú, Botrytis cinerea	2024-11-12 15:06:15.570159
304	12	Botrytis cinerea, Mildiú	2024-11-12 15:08:41.594227
305	12	Mildiú, Botrytis cinerea	2024-11-12 15:14:22.020707
306	12	Botrytis cinerea, Mildiú	2024-11-12 15:17:31.4183
307	12	Mildiú, Botrytis cinerea	2024-11-12 15:21:27.364018
308	12	Mildiú, Botrytis cinerea	2024-11-12 15:28:44.060879
309	12	Botrytis cinerea, Mildiú	2024-11-12 15:32:50.099075
310	12	Mildiú, Botrytis cinerea	2024-11-12 15:36:30.722886
311	12	Botrytis cinerea, Mildiú	2024-11-12 15:41:39.934535
312	12	Botrytis cinerea, Mildiú	2024-11-12 15:45:36.436932
313	12	Oídio, Saludable, Mildiú	2024-11-12 15:45:45.923727
314	12	Saludable, Mildiú, Oídio	2024-11-12 15:48:58.491333
315	12	Botrytis cinerea, Mildiú	2024-11-12 15:54:43.392209
316	12	Saludable, Mildiú, Oídio	2024-11-12 15:59:09.878964
317	12	Mildiú, Botrytis cinerea	2024-11-12 16:06:42.60444
318	12	Mildiú, Botrytis cinerea	2024-11-12 16:25:58.346761
319	12	Botrytis cinerea, Mildiú	2024-11-12 16:29:08.008554
320	12	Mildiú, Botrytis cinerea	2024-11-12 16:32:34.691631
321	12	Saludable, Oídio, Mildiú	2024-11-12 16:33:09.587066
322	12	Mildiú, Botrytis cinerea	2024-11-12 16:39:04.908234
323	12	Mildiú, Botrytis cinerea	2024-11-12 16:43:13.597016
324	12	Mildiú, Botrytis cinerea	2024-11-12 16:45:52.972517
325	12	Mildiú, Botrytis cinerea	2024-11-12 16:49:34.249512
326	12	Botrytis cinerea, Mildiú	2024-11-12 16:52:25.823721
327	12	Mildiú, Botrytis cinerea	2024-11-12 16:57:37.92994
328	12	Botrytis cinerea, Mildiú	2024-11-12 17:01:47.033498
329	12	Mildiú, Saludable, Oídio	2024-11-12 17:02:46.438706
330	12	Mildiú, Saludable, Oídio	2024-11-12 17:03:54.331119
331	12	Botrytis cinerea, Mildiú	2024-11-12 17:05:27.592193
332	12	Mildiú, Botrytis cinerea	2024-11-12 17:10:37.48881
333	12	Mildiú, Botrytis cinerea	2024-11-12 17:15:33.275116
334	12	Botrytis cinerea, Mildiú	2024-11-12 17:17:43.585162
335	12	Mildiú, Botrytis cinerea	2024-11-12 17:22:28.929307
336	12	Mildiú, Botrytis cinerea	2024-11-12 17:26:03.985659
337	12	Mildiú, Botrytis cinerea	2024-11-12 17:29:26.461573
338	12	Mildiú, Botrytis cinerea	2024-11-12 17:31:51.469147
339	12	Botrytis cinerea, Mildiú	2024-11-12 17:34:51.928616
340	12	Botrytis cinerea, Mildiú	2024-11-12 17:38:00.196537
341	12	Botrytis cinerea, Mildiú	2024-11-12 17:46:39.123905
342	12	Mildiú, Botrytis cinerea	2024-11-12 17:51:43.01126
343	12	Botrytis cinerea, Mildiú	2024-11-12 17:52:52.911318
344	12	Oídio, Saludable, Mildiú	2024-11-12 18:11:08.726798
345	12	Mildiú	2024-11-12 18:19:42.192294
346	12	Oídio	2024-11-12 18:19:42.271704
347	12	Mildiú	2024-11-12 18:19:42.3959
348	12	Mildiú	2024-11-12 18:19:42.603891
349	12	Botrytis cinerea	2024-11-12 18:23:01.439845
350	12	Mildiú	2024-11-12 18:23:10.241624
351	12	Oídio	2024-11-12 18:23:10.312948
352	12	Mildiú	2024-11-12 18:23:10.42971
353	12	Mildiú	2024-11-12 18:23:10.497701
354	12	Mildiú	2024-11-12 18:23:10.565841
355	12	Mildiú	2024-11-12 18:29:33.201814
356	12	Mildiú	2024-11-12 18:29:33.39358
357	12	Oídio	2024-11-12 18:29:33.477942
358	12	Mildiú	2024-11-12 18:29:33.606355
359	12	Mildiú	2024-11-12 18:29:33.690921
360	12	Mildiú	2024-11-12 18:29:33.770139
361	12	Mildiú	2024-11-12 18:41:29.380988
362	12	Mildiú	2024-11-12 18:41:29.457654
363	12	Oídio	2024-11-12 18:41:29.524434
364	12	Mildiú	2024-11-12 18:41:29.65415
365	12	Mildiú	2024-11-12 18:41:29.741134
366	12	Oídio, Saludable, Mildiú	2024-11-12 18:47:58.590992
367	12	Oídio, Saludable, Mildiú	2024-11-12 18:50:34.73638
368	12	Mildiú, Botrytis cinerea	2024-11-12 18:52:39.775506
369	12	Oídio, Saludable, Mildiú	2024-11-12 19:05:34.943456
370	12	Esca	2024-11-19 10:47:36.795517
371	12	Botrytis cinerea	2024-11-19 10:47:42.795472
372	12	Botrytis cinerea	2024-11-19 10:47:47.377473
373	12	Esca	2024-11-19 10:47:53.110642
374	12	Mildiú	2024-11-19 10:48:01.109466
375	12	Botrytis cinerea	2024-11-19 10:48:09.571915
376	12	Botrytis cinerea	2024-11-19 10:49:08.899657
377	12	Botrytis cinerea	2024-11-19 10:49:28.063963
378	12	Mildiú	2024-11-19 10:50:14.05566
379	12	Mildiú	2024-11-19 10:50:50.3304
380	12	Esca	2024-11-19 10:54:24.64104
381	12	Podredumbre negra	2024-11-19 11:24:38.877242
382	12	Oídio	2024-11-19 11:25:17.723869
383	12	Yesca	2024-11-19 11:25:23.135217
384	12	Yesca	2024-11-19 11:25:26.769105
385	12	Oídio	2024-11-19 11:25:36.348625
386	12	Podredumbre negra	2024-11-19 11:26:05.399635
387	12	Desconocida	2024-11-19 11:26:29.287006
388	12	Podredumbre negra	2024-11-19 11:26:39.744346
389	12	Desconocida	2024-11-19 11:27:57.91127
390	12	Yesca, Desconocida, Podredumbre negra	2024-11-19 11:29:43.384214
391	12	Podredumbre negra	2024-11-19 11:41:57.663685
392	12	Yesca	2024-11-19 11:44:19.592099
393	12	Yesca	2024-11-19 11:45:33.929673
394	12	Oídio	2024-11-19 11:45:40.227816
395	12	Tizon de la hoja	2024-11-19 11:46:40.243357
396	12	Oídio	2024-11-19 11:46:49.752669
397	12	Oídio	2024-11-19 11:46:54.714763
398	12	Oídio	2024-11-19 11:46:58.6444
399	12	Oídio	2024-11-19 11:47:00.978735
400	12	Oídio	2024-11-19 11:47:04.187925
401	12	Oídio	2024-11-19 11:47:09.057629
402	12	Oídio	2024-11-19 11:47:12.252051
403	12	Oídio	2024-11-19 11:47:15.167755
404	12	Otros	2024-11-19 11:47:50.508589
405	12	Otros	2024-11-19 11:47:55.266402
406	12	Otros	2024-11-19 11:47:59.421224
407	12	Otros	2024-11-19 11:48:03.27865
408	12	Saludable, Mildiú, Podredumbre negra, Yesca	2024-11-19 12:47:51.666467
409	12	Podredumbre negra	2024-11-19 13:44:51.61564
410	12	Podredumbre negra	2024-11-19 13:57:28.952584
411	12	Yesca	2024-11-19 14:13:41.644754
412	12	Yesca	2024-11-19 14:13:41.799314
413	12	Mildiú	2024-11-19 14:13:41.876852
414	12	Podredumbre negra	2024-11-19 14:13:42.033803
415	12	Oídio	2024-11-19 14:13:42.08608
416	12	Yesca	2024-11-19 14:16:36.861529
417	12	Yesca	2024-11-19 14:16:36.932765
418	12	Mildiú	2024-11-19 14:16:37.005692
419	12	Podredumbre negra	2024-11-19 14:16:37.052784
420	12	Oídio	2024-11-19 14:16:37.10199
421	12	Yesca	2024-11-19 14:19:42.873523
422	12	Yesca	2024-11-19 14:32:45.384654
423	12	Yesca	2024-11-19 14:32:45.449106
424	12	Mildiú	2024-11-19 14:32:45.53137
425	12	Podredumbre negra	2024-11-19 14:32:45.692813
426	12	Oídio	2024-11-19 14:32:45.746437
427	12	Podredumbre negra	2024-11-19 14:34:40.009191
428	12	Yesca	2024-11-19 14:38:44.178317
429	12	Yesca	2024-11-19 14:38:44.25826
430	12	Mildiú	2024-11-19 14:38:44.336252
431	12	Podredumbre negra	2024-11-19 14:38:44.385086
432	12	Oídio	2024-11-19 14:38:44.439865
433	12	Yesca	2024-11-19 14:41:09.472276
434	12	Yesca	2024-11-19 14:41:09.534587
435	12	Mildiú	2024-11-19 14:41:09.613133
436	12	Podredumbre negra	2024-11-19 14:41:09.660136
437	12	Oídio	2024-11-19 14:41:09.711661
438	12	Podredumbre negra	2024-11-19 14:41:09.761091
439	12	Yesca	2024-11-19 14:43:28.532547
440	12	Yesca	2024-11-19 14:43:28.588206
441	12	Mildiú	2024-11-19 14:43:28.655478
442	12	Podredumbre negra	2024-11-19 14:43:28.697411
443	12	Oídio	2024-11-19 14:43:28.743235
444	12	Yesca	2024-11-19 14:43:38.765879
445	12	Yesca	2024-11-19 14:43:38.837718
446	12	Mildiú	2024-11-19 14:43:38.912724
447	12	Podredumbre negra	2024-11-19 14:43:38.958472
448	12	Oídio	2024-11-19 14:43:39.011247
449	12	Podredumbre negra	2024-11-19 14:47:12.410707
450	12	Podredumbre negra	2024-11-19 14:50:08.274138
451	12	Yesca	2024-11-19 14:51:13.475155
452	12	Yesca	2024-11-19 14:51:13.542685
453	12	Mildiú	2024-11-19 14:51:13.610973
454	12	Podredumbre negra	2024-11-19 14:51:13.655581
455	12	Oídio	2024-11-19 14:51:13.709134
456	12	Yesca	2024-11-19 14:54:46.04172
457	12	Yesca	2024-11-19 14:54:46.099871
458	12	Mildiú	2024-11-19 14:54:46.175558
459	12	Podredumbre negra	2024-11-19 14:54:46.22024
460	12	Podredumbre negra	2024-11-19 14:54:55.820733
461	12	Yesca	2024-11-19 14:57:01.420592
462	12	Yesca	2024-11-19 14:57:01.486972
463	12	Mildiú	2024-11-19 14:57:01.558954
464	12	Podredumbre negra	2024-11-19 14:57:01.607308
465	12	Yesca	2024-11-19 14:57:01.654798
466	12	Podredumbre negra	2024-11-19 14:57:28.322779
467	12	Yesca	2024-11-19 15:00:40.623525
468	12	Yesca	2024-11-19 15:00:40.790854
469	12	Mildiú	2024-11-19 15:00:40.856433
470	12	Podredumbre negra	2024-11-19 15:00:40.897956
471	12	Yesca	2024-11-19 15:00:40.940719
472	12	Podredumbre negra	2024-11-19 15:01:00.65675
473	12	Podredumbre negra	2024-11-19 15:01:22.616345
474	12	Podredumbre negra	2024-11-19 15:02:50.732728
475	12	Yesca	2024-11-19 15:06:51.072901
476	12	Yesca	2024-11-19 15:06:51.130213
477	12	Mildiú	2024-11-19 15:06:51.199807
478	12	Podredumbre negra	2024-11-19 15:06:51.244315
479	12	Podredumbre negra	2024-11-19 15:07:13.481615
480	12	Podredumbre negra	2024-11-19 15:15:42.17323
481	12	Podredumbre negra	2024-11-19 15:31:20.357854
482	12	Podredumbre negra	2024-11-19 15:36:45.583446
483	12	Podredumbre negra	2024-11-19 15:48:01.118222
484	12	Podredumbre negra	2024-11-19 15:51:12.36538
485	12	Podredumbre negra	2024-11-19 15:51:12.431979
486	12	Podredumbre negra	2024-11-19 15:51:12.505596
487	12	Podredumbre negra	2024-11-19 15:51:12.550468
488	12	Podredumbre negra	2024-11-19 15:51:12.597226
489	12	Otros	2024-11-19 16:02:46.543308
490	12	Otros	2024-11-19 16:03:00.493046
491	12	Botrytis cinerea	2024-11-19 16:03:31.054777
492	12	Podredumbre negra	2024-11-19 16:04:57.64679
493	12	Podredumbre negra	2024-11-19 16:08:44.912965
494	12	Podredumbre negra	2024-11-19 16:08:44.970491
495	12	Podredumbre negra	2024-11-19 16:08:45.040998
496	12	Podredumbre negra	2024-11-19 16:08:45.086766
497	12	Botrytis cinerea	2024-11-19 16:08:45.130184
498	12	Yesca	2024-11-19 16:14:04.763087
499	12	Yesca	2024-11-19 16:14:04.823328
500	12	Mildiú	2024-11-19 16:14:04.894404
501	12	Podredumbre negra	2024-11-19 16:14:04.936219
502	12	Oídio	2024-11-19 16:14:04.984464
503	12	Yesca	2024-11-19 16:14:05.027388
504	12	Yesca	2024-11-19 16:19:40.965226
505	12	Yesca	2024-11-19 16:19:41.037526
506	12	Mildiú	2024-11-19 16:19:41.126455
507	12	Podredumbre negra	2024-11-19 16:19:41.17098
508	12	Oídio	2024-11-19 16:19:41.222666
509	12	Yesca	2024-11-19 16:19:41.268368
510	12	Podredumbre negra	2024-11-19 16:23:37.430568
511	12	Yesca	2024-11-19 16:25:53.925547
512	12	Yesca	2024-11-19 16:25:54.004317
513	12	Mildiú	2024-11-19 16:25:54.09927
514	12	Podredumbre negra	2024-11-19 16:25:54.162647
515	12	Yesca	2024-11-19 16:25:54.326101
516	12	Yesca	2024-11-19 16:29:07.822935
517	12	Yesca	2024-11-19 16:29:07.888506
518	12	Mildiú	2024-11-19 16:29:07.959955
519	12	Podredumbre negra	2024-11-19 16:29:08.006174
520	12	Oídio	2024-11-19 16:29:08.053263
521	12	Yesca	2024-11-19 16:29:08.098908
522	12	Yesca	2024-11-19 16:32:47.259042
523	12	Yesca	2024-11-19 16:32:47.314934
524	12	Mildiú	2024-11-19 16:32:47.388025
525	12	Podredumbre negra	2024-11-19 16:32:47.429794
526	12	Oídio	2024-11-19 16:32:47.477225
527	12	Yesca	2024-11-19 16:32:47.522854
528	12	Yesca	2024-11-19 16:37:14.860615
529	12	Yesca	2024-11-19 16:37:14.927858
530	12	Mildiú	2024-11-19 16:37:14.998053
531	12	Podredumbre negra	2024-11-19 16:37:15.043708
532	12	Oídio	2024-11-19 16:37:15.09366
533	12	Yesca	2024-11-19 16:37:15.135674
534	12	Yesca	2024-11-19 16:40:11.614238
535	12	Yesca	2024-11-19 16:40:11.684157
536	12	Mildiú	2024-11-19 16:40:11.765234
537	12	Podredumbre negra	2024-11-19 16:40:11.809558
538	12	Oídio	2024-11-19 16:40:11.858085
539	12	Yesca	2024-11-19 16:40:11.90371
540	12	Yesca	2024-11-19 16:43:41.43075
541	12	Yesca	2024-11-19 16:43:41.502182
542	12	Mildiú	2024-11-19 16:43:41.576802
543	12	Podredumbre negra	2024-11-19 16:43:41.618938
544	12	Oídio	2024-11-19 16:43:41.775671
545	12	Yesca	2024-11-19 16:43:41.820622
546	12	Yesca	2024-11-19 16:52:45.663363
547	12	Yesca	2024-11-19 16:52:45.729629
548	12	Mildiú	2024-11-19 16:52:45.818474
549	12	Podredumbre negra	2024-11-19 16:52:45.877407
550	12	Oídio	2024-11-19 16:52:45.940336
551	12	Podredumbre negra	2024-11-19 16:57:11.391747
552	12	Podredumbre negra	2024-11-19 17:02:06.298597
553	12	Yesca	2024-11-19 17:05:07.985512
554	12	Yesca	2024-11-19 17:05:08.059379
555	12	Mildiú	2024-11-19 17:05:08.136652
556	12	Podredumbre negra	2024-11-19 17:05:08.183081
557	12	Oídio	2024-11-19 17:05:08.234483
558	12	Yesca	2024-11-19 17:07:01.366809
559	12	Yesca	2024-11-19 17:07:01.435962
560	12	Mildiú	2024-11-19 17:07:01.619829
561	12	Podredumbre negra	2024-11-19 17:07:01.665458
562	12	Oídio	2024-11-19 17:07:01.713602
563	12	Yesca	2024-11-19 17:11:35.00305
564	12	Yesca	2024-11-19 17:11:35.078458
565	12	Mildiú	2024-11-19 17:11:35.156976
566	12	Podredumbre negra	2024-11-19 17:11:35.205338
567	12	Yesca	2024-11-19 17:11:35.253363
568	12	Yesca	2024-11-19 17:33:50.255885
569	12	Yesca	2024-11-19 17:33:50.314599
570	12	Mildiú	2024-11-19 17:33:50.385639
571	12	Podredumbre negra	2024-11-19 17:33:50.42948
572	12	Yesca	2024-11-19 17:33:50.472649
573	12	Yesca	2024-11-19 17:38:03.617924
574	12	Yesca	2024-11-19 17:38:03.685785
575	12	Mildiú	2024-11-19 17:38:03.760046
576	12	Podredumbre negra	2024-11-19 17:38:03.805391
577	12	Oídio	2024-11-19 17:38:03.855433
578	12	Yesca	2024-11-19 17:38:03.901388
579	12	Yesca	2024-11-19 17:40:46.631768
580	12	Yesca	2024-11-19 17:40:46.694079
581	12	Mildiú	2024-11-19 17:40:46.879961
582	12	Podredumbre negra	2024-11-19 17:40:46.927615
583	12	Oídio	2024-11-19 17:40:46.982408
584	12	Yesca	2024-11-19 17:40:47.02816
585	12	Yesca	2024-11-19 17:41:16.190025
586	12	Yesca	2024-11-19 17:41:16.246614
587	12	Mildiú	2024-11-19 17:41:16.320255
588	12	Podredumbre negra	2024-11-19 17:41:16.372318
589	12	Oídio	2024-11-19 17:41:16.427892
590	12	Yesca	2024-11-20 07:09:35.14226
591	12	Yesca	2024-11-20 07:09:35.31237
592	12	Mildiú	2024-11-20 07:09:35.383957
593	12	Podredumbre negra	2024-11-20 07:09:35.434608
594	12	Oídio	2024-11-20 07:09:35.481893
595	12	Yesca	2024-11-20 07:12:54.731926
596	12	Yesca	2024-11-20 07:12:54.797796
597	12	Mildiú	2024-11-20 07:12:54.87567
598	12	Podredumbre negra	2024-11-20 07:12:54.926146
599	12	Oídio	2024-11-20 07:12:55.081525
600	12	Yesca	2024-11-20 07:16:00.085732
601	12	Yesca	2024-11-20 07:16:00.151867
602	12	Mildiú	2024-11-20 07:16:00.229689
603	12	Podredumbre negra	2024-11-20 07:16:00.385734
604	12	Oídio	2024-11-20 07:16:00.433095
605	12	Yesca	2024-11-20 07:16:00.47981
606	12	Yesca	2024-11-20 07:18:49.272445
607	12	Yesca	2024-11-20 07:18:49.452766
608	12	Mildiú	2024-11-20 07:18:49.642034
609	12	Podredumbre negra	2024-11-20 07:18:49.682115
610	12	Oídio	2024-11-20 07:18:49.728707
611	12	Podredumbre negra	2024-11-20 07:22:15.437057
612	12	Yesca	2024-11-20 07:22:15.494105
613	12	Podredumbre negra	2024-11-20 07:22:49.703916
614	12	Saludable, Yesca, Mildiú, Podredumbre negra	2024-11-20 07:24:07.017381
615	12	Podredumbre negra	2024-11-20 07:26:41.397023
616	12	Podredumbre negra	2024-11-20 07:29:59.255653
617	12	Yesca, Botrytis cinerea, Mildiú, Podredumbre negra	2024-11-20 07:37:46.346182
618	12	Podredumbre negra	2024-11-20 07:41:18.48246
619	12	Yesca, Saludable, Podredumbre negra, Mildiú	2024-11-20 07:47:24.507757
620	12	Yesca	2024-11-20 07:47:40.313673
621	12	Yesca	2024-11-20 07:47:40.495601
622	12	Mildiú	2024-11-20 07:47:40.695693
623	12	Podredumbre negra	2024-11-20 07:47:40.743934
624	12	Oídio	2024-11-20 07:47:40.802598
625	12	Yesca	2024-11-20 07:56:56.943905
626	12	Mildiú	2024-11-20 07:56:57.014701
627	12	Podredumbre negra	2024-11-20 07:56:57.057269
628	12	Oídio	2024-11-20 07:56:57.204724
629	12	Podredumbre negra	2024-11-20 07:57:07.977773
630	12	Podredumbre negra	2024-11-20 08:01:05.288234
631	12	Podredumbre negra	2024-11-20 08:03:57.274454
632	12	Podredumbre negra	2024-11-20 08:08:43.682549
633	12	Podredumbre negra	2024-11-20 08:09:16.397278
634	12	Yesca	2024-11-20 08:09:32.366027
635	12	Yesca	2024-11-20 08:09:32.533661
636	12	Mildiú	2024-11-20 08:09:32.610374
637	12	Podredumbre negra	2024-11-20 08:09:32.65547
638	12	Oídio	2024-11-20 08:09:32.705983
639	12	Yesca	2024-11-21 00:31:42.931854
640	12	Yesca	2024-11-21 00:31:42.997128
641	12	Mildiú	2024-11-21 00:31:43.079709
642	12	Podredumbre negra	2024-11-21 00:31:43.130724
643	12	Oídio	2024-11-21 00:31:43.181168
644	12	Podredumbre negra	2024-11-21 00:31:43.229585
645	12	Podredumbre negra	2024-11-21 00:31:43.278531
646	12	Yesca	2024-11-21 00:31:43.324847
647	12	Podredumbre negra	2024-11-21 00:32:09.310563
648	12	Podredumbre negra	2024-11-21 00:32:40.887733
649	12	Podredumbre negra, Yesca, Saludable	2024-11-21 00:33:48.691555
\.


--
-- TOC entry 4860 (class 0 OID 16412)
-- Dependencies: 215
-- Data for Name: usuarios; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.usuarios (id, nombre, correo, "contraseña", fecha_registro, estado, rol) FROM stdin;
1	Henry	gato_210191@hotmail.com	1234	2024-10-12 22:21:12.879781	t	user
3	jenry	j@gmail.com	12345	2024-10-12 22:44:56.167688	t	user
4	Gabriel	g@gmail.com	1234	2024-10-12 23:06:05.675679	t	user
5	Henry	gato_2101@hotmail.com	12345	2024-10-12 23:29:18.488972	t	user
6	Yoleida	yoleida@gmail.com	Yoleida12345	2024-10-13 07:28:18.499925	t	user
7	Maria	henry.castro@example.com	Henry12345	2024-10-13 18:48:42.451319	t	user
8	Joseph	joseph@gmail.com	Joseph1234	2024-10-14 07:28:57.741383	t	user
9	Carmen	carmen@gmail.com	Carmen1234	2024-10-14 10:10:10.729676	t	user
10	Joseph	josephg@gmail.com	Joseph1234	2024-10-14 10:20:20.334553	t	user
11	Yoleida	yoleida1@gmail.com	Yoleida12345	2024-10-14 11:32:04.377516	t	user
12	Jesus De la cruz	jesuscruz23@gmail.com	Hola1234	2024-10-15 13:03:16.005676	t	user
13	luchomiguel	luismiguel@gmail.com	scrypt:32768:8:1$Un8zKaQ7cfGtmDje$43d56b2294f367878dad885b37c8cab7c5c09f8884ece4abc9325eb8637310047a3dbe761e35fbc5fdf451cd3c8f77329363f61394ae4d579284265750717817	2024-10-28 11:16:32.801848	t	user
14	Luis Morales	luchomoranpi93@gmail.com	Hola1234	2024-11-05 11:41:16.317131	t	user
15	.d	loquitodelacalle34@gmail.com	Tebrinco5	2024-11-11 11:40:53.797575	t	user
16	Juan	loquitodelacalle35@gmail.com	Tebrinco5	2024-11-11 11:58:53.147455	t	user
17	.#pdiddy6	pedede23@gmail.com	Hola1234	2024-11-19 09:23:01.264772	t	user
18	Administrador	vanguardgicen@gmail.com	Hola1234	2024-11-21 13:50:56.776966	t	admin
\.


--
-- TOC entry 4865 (class 0 OID 16578)
-- Dependencies: 220
-- Data for Name: validaciones; Type: TABLE DATA; Schema: public; Owner: usuario_uva
--

COPY public.validaciones (id, user_id, nombre_usuario, frame_path, user_class, model_class, fecha_validacion) FROM stdin;
1	12	Jesus De la cruz	uploads/IMG_3073.JPEG	Otros	Yesca	2024-11-21 13:01:55.425855
2	12	Jesus De la cruz	uploads/IMG_3250.JPEG	Otros	Yesca	2024-11-21 13:01:55.425855
3	12	Jesus De la cruz	uploads/IMG_3226.JPEG	Otros	Mildiú	2024-11-21 13:01:55.425855
4	12	Jesus De la cruz	uploads/WhatsApp Image 2024-11-19 at 11.26.18 AM.jpeg	Otros	Podredumbre negra	2024-11-21 13:01:55.425855
5	12	Jesus De la cruz	uploads/frame_10.png	Tizón de la hoja	Saludable	2024-11-21 13:06:34.874505
6	12	Jesus De la cruz	uploads/frame_60.png	Tizón de la hoja	Saludable	2024-11-21 13:06:34.874505
7	12	Jesus De la cruz	uploads/frame_90.png	Tizón de la hoja	Yesca	2024-11-21 13:06:34.874505
8	12	Jesus De la cruz	uploads/frame_140.png	Tizón de la hoja	Yesca	2024-11-21 13:06:34.874505
9	12	Jesus De la cruz	uploads/frame_250.png	Tizón de la hoja	Mildiú	2024-11-21 13:06:34.874505
10	9	Jesus De la cruz	uploads/images (6).jpg	Oídio	Yesca	2024-11-22 08:41:28.515496
11	9	Jesus De la cruz	uploads/images (5).jpg	Oídio	Yesca	2024-11-22 08:41:28.515496
12	12	Jesus De la cruz	uploads/AdobeStock_293135869-compressed.jpg	Botrytis cinerea	Yesca	2024-11-22 08:52:56.01251
13	12	Jesus De la cruz	uploads/images (1).jpg	Esca	Yesca	2024-11-22 08:52:56.01251
14	12	Jesus De la cruz	uploads/images.jpg	Mildiú	Oídio	2024-11-22 08:52:56.01251
15	12	Jesus De la cruz	uploads/esca-grape-1665660551.jpg	Oídio	Podredumbre negra	2024-11-22 08:52:56.01251
16	12	Jesus De la cruz	uploads/Vid-ritec-min-1.webp	Podredumbre negra	Yesca	2024-11-22 08:52:56.01251
17	12	Jesus De la cruz	uploads/frame_50.png	Oídio	Yesca	2024-11-22 08:54:27.283955
18	12	Jesus De la cruz	uploads/frame_100.png	Podredumbre negra	Yesca	2024-11-22 08:54:27.283955
19	12	Jesus De la cruz	uploads/frame_170.png	Otros	Yesca	2024-11-22 08:54:27.283955
20	10	Jesus De la cruz	uploads/tizon-de-la-hoja-Cercospora-kikuchii.jpg	Botrytis cinerea	Oídio	2024-11-22 09:02:29.584279
21	10	Jesus De la cruz	uploads/images (3).jpg	Oídio	Botrytis cinerea	2024-11-22 09:02:29.584279
22	12	Jesus De la cruz	uploads/images (9).jpg	Botrytis cinerea	Oídio	2024-11-22 11:31:43.264257
23	12	Jesus De la cruz	uploads/images (6).jpg	Oídio	Yesca	2024-11-22 11:31:43.264257
24	12	Jesus De la cruz	uploads/frame_130.png	Oídio	Podredumbre negra	2024-11-22 11:32:40.276591
25	12	Jesus De la cruz	uploads/frame_170.png	Mildiú	Podredumbre negra	2024-11-22 11:32:40.276591
26	12	Jesus De la cruz	uploads/IMG_3250.JPEG	Oídio	Yesca	2024-11-22 12:15:00.148611
27	12	Jesus De la cruz	uploads/IMG_3073.JPEG	Tizón de la hoja	Yesca	2024-11-22 12:15:00.148611
28	12	Jesus De la cruz	uploads/IMG_3226.JPEG	Botrytis cinerea	Mildiú	2024-11-22 12:15:00.148611
29	12	Jesus De la cruz	uploads/WhatsApp Image 2024-11-19 at 11.26.18 AM.jpeg	Otros	Podredumbre negra	2024-11-22 18:40:39.742019
30	12	Jesus De la cruz	uploads/images (9).jpg	Mildiú	Podredumbre negra	2024-11-22 18:40:39.742019
31	12	Jesus De la cruz	uploads/images (8).jpg	Otros	Otros	2024-11-22 18:40:39.742019
32	12	Jesus De la cruz	uploads/images (7).jpg	Otros	Otros	2024-11-22 18:40:39.742019
33	12	Jesus De la cruz	uploads/images (7).jpg	Otros	Otros	2024-11-22 19:23:24.104472
34	12	Jesus De la cruz	uploads/images (8).jpg	Otros	Otros	2024-11-22 19:23:24.104472
\.


--
-- TOC entry 4877 (class 0 OID 0)
-- Dependencies: 217
-- Name: consultas_id_seq; Type: SEQUENCE SET; Schema: public; Owner: usuario_uva
--

SELECT pg_catalog.setval('public.consultas_id_seq', 649, true);


--
-- TOC entry 4878 (class 0 OID 0)
-- Dependencies: 216
-- Name: usuarios_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.usuarios_id_seq', 18, true);


--
-- TOC entry 4879 (class 0 OID 0)
-- Dependencies: 219
-- Name: validaciones_id_seq; Type: SEQUENCE SET; Schema: public; Owner: usuario_uva
--

SELECT pg_catalog.setval('public.validaciones_id_seq', 34, true);


--
-- TOC entry 4712 (class 2606 OID 16455)
-- Name: consultas consultas_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario_uva
--

ALTER TABLE ONLY public.consultas
    ADD CONSTRAINT consultas_pkey PRIMARY KEY (id);


--
-- TOC entry 4708 (class 2606 OID 16430)
-- Name: usuarios usuarios_correo_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.usuarios
    ADD CONSTRAINT usuarios_correo_key UNIQUE (correo);


--
-- TOC entry 4710 (class 2606 OID 16432)
-- Name: usuarios usuarios_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.usuarios
    ADD CONSTRAINT usuarios_pkey PRIMARY KEY (id);


--
-- TOC entry 4714 (class 2606 OID 16586)
-- Name: validaciones validaciones_pkey; Type: CONSTRAINT; Schema: public; Owner: usuario_uva
--

ALTER TABLE ONLY public.validaciones
    ADD CONSTRAINT validaciones_pkey PRIMARY KEY (id);


--
-- TOC entry 4715 (class 2606 OID 16456)
-- Name: consultas consultas_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario_uva
--

ALTER TABLE ONLY public.consultas
    ADD CONSTRAINT consultas_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usuarios(id);


--
-- TOC entry 4716 (class 2606 OID 16587)
-- Name: validaciones validaciones_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: usuario_uva
--

ALTER TABLE ONLY public.validaciones
    ADD CONSTRAINT validaciones_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usuarios(id);


--
-- TOC entry 4871 (class 0 OID 0)
-- Dependencies: 5
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: pg_database_owner
--

GRANT ALL ON SCHEMA public TO usuario_uva;


--
-- TOC entry 4873 (class 0 OID 0)
-- Dependencies: 215
-- Name: TABLE usuarios; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.usuarios TO usuario_uva;


--
-- TOC entry 4875 (class 0 OID 0)
-- Dependencies: 216
-- Name: SEQUENCE usuarios_id_seq; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT,USAGE ON SEQUENCE public.usuarios_id_seq TO usuario_uva;


-- Completed on 2024-11-24 15:14:03

--
-- PostgreSQL database dump complete
--

