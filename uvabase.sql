PGDMP      
            
    |            proyect_uva    16.4    16.4                0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false                       0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false                       0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false                       1262    16403    proyect_uva    DATABASE     �   CREATE DATABASE proyect_uva WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'English_United States.1252';
    DROP DATABASE proyect_uva;
                usuario_uva    false                       0    0    SCHEMA public    ACL     +   GRANT ALL ON SCHEMA public TO usuario_uva;
                   pg_database_owner    false    5            �            1259    16449 	   consultas    TABLE     �   CREATE TABLE public.consultas (
    id integer NOT NULL,
    user_id integer,
    class_name character varying(100),
    consulta_fecha timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);
    DROP TABLE public.consultas;
       public         heap    usuario_uva    false            �            1259    16448    consultas_id_seq    SEQUENCE     �   CREATE SEQUENCE public.consultas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 '   DROP SEQUENCE public.consultas_id_seq;
       public          usuario_uva    false    218            	           0    0    consultas_id_seq    SEQUENCE OWNED BY     E   ALTER SEQUENCE public.consultas_id_seq OWNED BY public.consultas.id;
          public          usuario_uva    false    217            �            1259    16412    usuarios    TABLE     �  CREATE TABLE public.usuarios (
    id integer NOT NULL,
    nombre character varying(255) NOT NULL,
    correo character varying(255) NOT NULL,
    "contraseña" character varying(255) NOT NULL,
    fecha_registro timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    estado boolean DEFAULT true,
    rol character varying(10) DEFAULT 'user'::character varying NOT NULL,
    CONSTRAINT chk_rol CHECK (((rol)::text = ANY ((ARRAY['user'::character varying, 'admin'::character varying])::text[])))
);
    DROP TABLE public.usuarios;
       public         heap    postgres    false            
           0    0    TABLE usuarios    ACL     K   GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE public.usuarios TO usuario_uva;
          public          postgres    false    215            �            1259    16419    usuarios_id_seq    SEQUENCE     �   CREATE SEQUENCE public.usuarios_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 &   DROP SEQUENCE public.usuarios_id_seq;
       public          postgres    false    215                       0    0    usuarios_id_seq    SEQUENCE OWNED BY     C   ALTER SEQUENCE public.usuarios_id_seq OWNED BY public.usuarios.id;
          public          postgres    false    216                       0    0    SEQUENCE usuarios_id_seq    ACL     F   GRANT SELECT,USAGE ON SEQUENCE public.usuarios_id_seq TO usuario_uva;
          public          postgres    false    216            �            1259    16578    validaciones    TABLE     t  CREATE TABLE public.validaciones (
    id integer NOT NULL,
    user_id integer NOT NULL,
    nombre_usuario character varying(255) NOT NULL,
    frame_path character varying(500) NOT NULL,
    user_class character varying(50) NOT NULL,
    model_class character varying(50) NOT NULL,
    fecha_validacion timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);
     DROP TABLE public.validaciones;
       public         heap    usuario_uva    false            �            1259    16577    validaciones_id_seq    SEQUENCE     �   CREATE SEQUENCE public.validaciones_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 *   DROP SEQUENCE public.validaciones_id_seq;
       public          usuario_uva    false    220                       0    0    validaciones_id_seq    SEQUENCE OWNED BY     K   ALTER SEQUENCE public.validaciones_id_seq OWNED BY public.validaciones.id;
          public          usuario_uva    false    219            ^           2604    16452    consultas id    DEFAULT     l   ALTER TABLE ONLY public.consultas ALTER COLUMN id SET DEFAULT nextval('public.consultas_id_seq'::regclass);
 ;   ALTER TABLE public.consultas ALTER COLUMN id DROP DEFAULT;
       public          usuario_uva    false    218    217    218            Z           2604    16422    usuarios id    DEFAULT     j   ALTER TABLE ONLY public.usuarios ALTER COLUMN id SET DEFAULT nextval('public.usuarios_id_seq'::regclass);
 :   ALTER TABLE public.usuarios ALTER COLUMN id DROP DEFAULT;
       public          postgres    false    216    215            `           2604    16581    validaciones id    DEFAULT     r   ALTER TABLE ONLY public.validaciones ALTER COLUMN id SET DEFAULT nextval('public.validaciones_id_seq'::regclass);
 >   ALTER TABLE public.validaciones ALTER COLUMN id DROP DEFAULT;
       public          usuario_uva    false    220    219    220            �          0    16449 	   consultas 
   TABLE DATA           L   COPY public.consultas (id, user_id, class_name, consulta_fecha) FROM stdin;
    public          usuario_uva    false    218   �#       �          0    16412    usuarios 
   TABLE DATA           b   COPY public.usuarios (id, nombre, correo, "contraseña", fecha_registro, estado, rol) FROM stdin;
    public          postgres    false    215   �@                 0    16578    validaciones 
   TABLE DATA           z   COPY public.validaciones (id, user_id, nombre_usuario, frame_path, user_class, model_class, fecha_validacion) FROM stdin;
    public          usuario_uva    false    220   cC                  0    0    consultas_id_seq    SEQUENCE SET     @   SELECT pg_catalog.setval('public.consultas_id_seq', 649, true);
          public          usuario_uva    false    217                       0    0    usuarios_id_seq    SEQUENCE SET     >   SELECT pg_catalog.setval('public.usuarios_id_seq', 18, true);
          public          postgres    false    216                       0    0    validaciones_id_seq    SEQUENCE SET     B   SELECT pg_catalog.setval('public.validaciones_id_seq', 34, true);
          public          usuario_uva    false    219            h           2606    16455    consultas consultas_pkey 
   CONSTRAINT     V   ALTER TABLE ONLY public.consultas
    ADD CONSTRAINT consultas_pkey PRIMARY KEY (id);
 B   ALTER TABLE ONLY public.consultas DROP CONSTRAINT consultas_pkey;
       public            usuario_uva    false    218            d           2606    16430    usuarios usuarios_correo_key 
   CONSTRAINT     Y   ALTER TABLE ONLY public.usuarios
    ADD CONSTRAINT usuarios_correo_key UNIQUE (correo);
 F   ALTER TABLE ONLY public.usuarios DROP CONSTRAINT usuarios_correo_key;
       public            postgres    false    215            f           2606    16432    usuarios usuarios_pkey 
   CONSTRAINT     T   ALTER TABLE ONLY public.usuarios
    ADD CONSTRAINT usuarios_pkey PRIMARY KEY (id);
 @   ALTER TABLE ONLY public.usuarios DROP CONSTRAINT usuarios_pkey;
       public            postgres    false    215            j           2606    16586    validaciones validaciones_pkey 
   CONSTRAINT     \   ALTER TABLE ONLY public.validaciones
    ADD CONSTRAINT validaciones_pkey PRIMARY KEY (id);
 H   ALTER TABLE ONLY public.validaciones DROP CONSTRAINT validaciones_pkey;
       public            usuario_uva    false    220            k           2606    16456     consultas consultas_user_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.consultas
    ADD CONSTRAINT consultas_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usuarios(id);
 J   ALTER TABLE ONLY public.consultas DROP CONSTRAINT consultas_user_id_fkey;
       public          usuario_uva    false    218    4710    215            l           2606    16587 &   validaciones validaciones_user_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.validaciones
    ADD CONSTRAINT validaciones_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usuarios(id);
 P   ALTER TABLE ONLY public.validaciones DROP CONSTRAINT validaciones_user_id_fkey;
       public          usuario_uva    false    215    220    4710            �      x��]��$9r�/��?`7@< �ȇH�H�f��݌�C�ۡ��	�?Q�F�~��@UNv7P����<2�D�G �����w����O_���K!�1���K̯Q^��Isʶ�8}U�Jڄ������_^>}~������?��(�!�R�˦+�WT�"�(�-I�k��W��n>��� {̩���8�{M�*[!�o�����~���˟�������P��j��>ry��U}�J-[O�Q^-�&�5����q8���ɫ�T��ӓ�@$�����1�e�����@+�
��L���)�"�h��E�<��������}�����*���g�5zZ�C�����K$I�1m���������P��<��&.�"�4�YZ�=�b�������|��?~���kMj�3g���vةa�K�2�J�c�0���^SPߒ�'釗	�N`W�,L�T8b���"�'C�/A^S�{�I R���@Xiٳ��e���lj9��T\7k�aq*1L\�d�=;�^��5' ǺP���Rb�g�k"�8�B-�Xy�t�R��^���/1��ĸ�b)�M�
��D�RU�M�f`|%��c��y�.✻PfD�6�q�5�n�+BgZ���<�`�δF �u�%��jy:a�Us��	QRu�h����7�)��L�����ׁ�pO%K�M���P+�y�Ysݴ<0���s���S8�z�`6U��f�-׌�cS�f���*4�8U�qҠ���7�gZӮ�q�ͦJC���,�%u3{�\f�K�Xi�=���b�u��j��f�<�fS�k@��ZKƊ��+\�!��4B��g����6068V_{J�A���H|	𰕦O��\O��9�*�dOP�?!�]D��^�(o�ėv���& tzS����?��|���*�.�Z��T��?�����>��?�~���7x��@g5���:��{p� �|��<���m^���#N��3lW��4�+%ղ������8l�Mw���ӷϟ���?~������6*fւ��!#`"B��G)a:fD�#�oٞ@1񉎹nٟAm�1$��@G-��sM�*�7��@�z�q���(�����i�[z���в������Q-��2t�7����3�#M)����Z��o~"����ʦˊ9f��|�N��!b��c	�\a:ϩ���"���z����"�yy�=@�juf�q�wH%r·�jd�`ts����
4YS��ճ&x����l�VK
J^��ޟ6xV{�4�9N���!8!Wܪ��d���c�(B!f%�y�3�B�jNϷ�$��"�t,BUO��w�^��2���2p� �`���E������)Bj������q"4�5��|`�$2�gS^ ma�C����������Wp8y�[3m,E0Rʔ7��0��j�JTw���3�S�=��ؚ*MN�H~`���cЖ�(�89&��`*N"F��P����HcI��Jr���H�-z�Q�8���"P;�}�X�M�gj�]�%1�`O%�Y�d�oG��4�^�y��dQ�>E�3^���])'�3x�H�w �(@FJ���*B'$�L,�\�#�К׌�Rj�#��4�e*�W� �і�8���*�A��4��Ɋ%�'ޑ������q@����Ud�B�R�>d����c�i�m�q���ȍ�@V�aP�a�e��;�@�*X$.3ށ�OXU�		�
��� �H�g��(S�ݑ`N�8*ݑX�%K�M�]Cb�&X�����Rd�\�F��30�=F��d� e*D��9B��c�r"�
�+@?��#A�{�E��CS`�0d-0�qh�X�%JL/�n��ՠ�;��m0K�u�svY'ɂ�U�'�
D����p�0��t��F��
���u��,�>q�������Gՠ3�/>'p�`���"#�M|Tz��K	Z����8������1�}/۫�����L�WV�u��W�N�Ɯw6D��KN�~�Ik�^�9�i�rzɅu�hsEZ+3"�hq�(H�@�<n�H�Hk� X����c<�P"��3oy �+ɳϼ�T@i{�He	+MJ䰆��p��)��b�)�c�Mi>�+�e�k軹��mj��?@�C��[�S�j}�]�H�T�������:i�m	\CI�@���=���	���pһ� ����5�n$C�w���@.N��/�ٓ$��P�L��*���m����{����;k�YD��M�a9b&�R۰Ip3��Q}�@ֶۗ�V�B9Y��.,���
��|��A2Vc>��:=Yi�ִ�*��A!� J�㪙1�L�,��!޸j�������S��Yo)��nw +.�\��M�yJ[G/�-'h�WV)�5����Y�/��dt*��܈.��k���?K�����P��4�R�*TZǎ���mն�W��P��]�w�����UG�28�������8,&2Њ�87�$nF+du�;�Y����m)Yq���s�a�=ì��!��	0��l
@�M�U�'�,JFw`�*c-����d�z$P.��f�a��E��&����p�F�!Y�	�NЁ�iFJ�C�ہ��o1�p �iƹ�I����y�Ű"�3�{p��L,2��z�&� �� q����OQ}L�m��=0du�{�[�B�,)ŉ��/�VV�%qr��1�����|�l��ܑ�t��w� �F֑H|�{J�M��sC��_�X������i�b���Q��H�Vp�����![B�1�-�Z~�H���>=w��Xi]�03?iȉn��0��1ȡ'�H�a�]���JbC��9W�Wi\��!�Y!�	�^�Vs�M��ͳ	��E��,mo*g�\�Q����rkA	�"�=di]%��C���[���=ܨL��<BB���a����{�;5�}��2��G|��X�y����U����#�I��D��	[��`[b����!�t�� �Q�OGJ3@6Z��Q������EZ-��t[E��o��p���+��;��He(�7G���/�.�v��u1i��V�d� h�]LQ�,�2��6�����n��A#+1����uYv�Yɂ�����T�5� ����٪�'����`�Iyl1�d�FJe��ꒁl�F"RP$�c/���;;nx	����!�h�������+�쉲�	P�E "�T6�Av5t�ք*�3����&��bۑ4m��*�m�TRNc�����l3v�����у[$��9�VQm\K��[�x|��e�+""h���º�M���C6�_ھo�07x��*��U_�n�ZnEa=s]�bQ��Z�>������b��P,�*�I^_W�m�d�_�j��˂&�P:� ���u�\S#JH�Ha�շ�\�V��'G
�@���UlG�[�$t�0�e9]�]H�$W�r7QmGK2�B,�qw�L6�R]tX.?w�]���ޤ�,�y�V�eY�V��{�b��&�3psW�w�zׅ�u�x�J�U��ey`�D�Mȩ\XW���A��Bf[��&S�Jn'�Lm�۬�
+�]�����௄-�Pp�9�x�j!��Ҕ�{�|�\XXrK;sQOxo�௄�(:�>�A����K�ω��T�����[d�U�֕6��#	!�y�P�e��8�B�^pWʾiI;]dq��َ�=���Vm�p�o��x:�E��2��K�)$E@� {!jk�g��e�	�U/�3�5�m�^����+l��h.�+c�egxA��^���x�mV!{�_�20{M��S�셅e- ���){�_Y����k���ݎY�+w %\�VFS�j�+�3uaU�28���xV)^XU��^����琽���`Uxh��V��"��W��r���0�oE�ɥ�V�f���p!���/�Ac%��m!�ҋ��I,��vR�����P�ȨR�����,�J_y��+�ۡ����C��+��*��l��y��n��gEai�7K�e���w��$`I8��Ys��3�!�����ל,bU����\;1n�.8���a{�5p�/�+�⮕Su!f�Ĕ�7!�   �w%�Y��!�Ov���k��,�V`:"|Vn�5��
�_.L7Y&�y�R
ϥ^XW�� ��\互�0���d10�^XW���1����Z��-��Ÿ9'ra]e:�D�2(r�a�Mx`x�/8��y��9�z!#dkI;�R��[�Ҳ��#���P/-+�B�*9��=���rb2�Q������i4�I�#[	;"��M�h'�L92��q��)�#�q���A�}-:\�D��q_�r7A���m�
��i�Ѯ���5wO���xhMl�y�̶��}��%�R�'N����3ރ<!���{F;�gw �g�yFzp���V�YCe�wE���!�2wEv$7�*+A����D�4>j�8-��F��"�H<����]�g`V�+y%�m5r�8ôU���%a����_pǥ]WPy���}�����0���~OgskW"��*�Z�N��A�C�"�t8���H��^���,����L�s���;���CU������Y�]b�r��+��E	/������I�����ұ��V �4k_0��;��JB�^fzdg48�Ɔjj��3�"�'J�әs�l�Q?s��D���$��z��_?�C��C}��\K�F:�9[�"C���� bq"J�(��������15�EϿ� ����O_>��	d�&�˜o�Y�;Q�>yCn�D6�K�g�9���$���ӗ����0H]R����܅� I�i�����3�9pS���ku�ӊ����v�B!��K��΄"�L(o;���!�9��}K�2�:�w r�*KK�����@n	2��2�㊓��9K���߁�G�� p�mN���T� ;-oF�h{�V��^���o߾��glO.V�f{�cãct��#B{�	tNo}�#\iS��^�q>,�^>��v�)�ct�t�W�ۄ��T�Ds�잉�S��h d��<Z�.�f#nbnw\�U"qc�zB�����j���v���@	�_h�\`q^X�j�Ѳ�tGVg\��v6�UI����5ȼ�`^�Uj;d,񼱦��݁1D�2Ms]�䆦�0Ki�+�9��V�Y5M\��ZyǍ�����&���Rճ��"�-�8��3�g`V�������v�E�2e���Ls��g��8�
rj��4������["�!S����Ǽ��r5�To�<Бy%p�r��t�׏O������%��qd[�,�݁��¡,Y�+���,Sf�8v�#ˑ^���'�R�����t��b��E�d*S5��B����u�9�g���;�k-s\ˏ�8�?!+��.Y�M$�Ƶ���JRMKα�D3-5eR��&a����M%�MY��HFaSG�q�T�{�:��,Rm)��d�x��<�u`�RG��6�j�Nr6mR6j7�$�{����d�󹤷.���M}Mm��<L/ӻ�\m�U�@c}���Cbǉ�U=#�u������d�%����p���З|fge�M���
y��K���$��q��Ҿo��ߑ���ByAo�L�<�y��;��BG|��;������3>��VD�aٳ�e�n����PR�(�u�a��x1��	�q�m���q����Y{�����2,!F~�=�&���x�0dʒb���`�B�Rw1�#�m+�,�X���"��I˒�o5�XR�m�%%��]��;��`�k��s�x�2�G�,��C��2K��&�F6z;��!/�ۦ��U<�Wi�8��i�,����"s�o�X�uA��n�;�Z�h]P����ۦJ�[�}��d�uA�]���1&BfAۇL��v�</��E�"�:���Df;��qv'���+I���4-9�X�\��LE*�m�B�,�b>|��t���O
�;!���X�7�o��qX�²pSr{G�,��=���X;�w\���\Y�E`�R��Ew�C@�5�I�����I	pb�q<֧Lwm^�9��&�����b�����.�f5�ݐ?_�٬Fs�Jᆒ�k4wd�ʋ_l�F�e���o��D���\WW��mV�i�N��d5hnjU'���7-��RH��Ti�2��|��Js s����T[��&O�Ye���!x,�-6���qX�<�i��́�2(h�4�e��:�Jx��f �%o��+����x��f���F������W4ln|����T�Ld�v����������p����pٰ�Y1�u�);.�j:���U=�w��7������%��e<���bg`��)�j��^�N���L�~�za�W�`��Ԓ:	7Zl��rB�t�R~9d2�����~�y�E��mɬ�v�|*�@[�_�=�������<:�bl}4��X��_Wi�O��wd�i����
�E1�9a�� ���܎�`��y�����8����X׏/(�q,s��30c�hq>�V;���ʟ�0�*�˥����:2�׌�Ru��$�����r�[��Ү� \���|���C�#�l|�d�L���.�����g��d�M5%m�4�g\,OUu�V�{�����wZ^RU�a�����]݀f� �\WWClII��J����8���I�:p,va:�wv��%�������]l����U]o�%)�ˬk��xڊ=V&�+�Kk��%�+��u�%uP��;R�Gr���!��S�̺�@)��>u��^��V�]Miu@4�0�����g��M����q�a�55�?y,d$ t�q�l�%��U]��՜+qC]����'�:�`��=l@�d%���a��خ�����금<���UZg$�K/�%euޔ�a��d�����\�C,a���~�/'U>n��8E��<�Ð����S�'���LI��ԇ�k���hPB|��n�r0���w��XR[j�s@���|8�uk]�w�\��&�̣��$�7�v����@�I����vu.(�Q��?6�^��xp1m�d�t��'�nb=�4�����SN!7޴�y�ڼ�� ����QI��m|�?g���;ҁ�|����2�阖��6߁�������y�E�v���W{}Gr���0�Y��0>�T%�g`
�OK�{�H�2�a��I�2������v~Ο�]n2�Ɩ��(��dڝ��N�p!���~uO����ź�o8
V�s�1A ��q=��o�e~�j�V��a9挃#��Zp��~AK�[x{��,��1op5ǄY7�|G
�4��3��d�����1o�%F�0���2� ��y��2�_Jj�8Iʟ��g-37�v�� v!�t}�|�2�+���2cC=���{BnO��Z<?�_��þm��O�@?      �   �  x���M��0��ʯ0t�+4�ɧ-]�e�=�C�PdIN;u��쯯�$�|�n�@��<~g�� y�M�#s�mq``�n�nW���oW���3.o���8�9���A�Ȗ����Y�����,T]�J�+MA�6�+Ƀ+�*�d~},r�s��F���ե�+���m�Jc,�#E�m���n��9��@��a�͈��ruD!yv]��b�E��l��.�u�uGި��&�iq*��3�����,��D����J� J!E	#ƒO�[ņ�q;���/1��qQ�6��#ع���z������� \ޭx�A.x�$�
^Uq�7�&��Y�2��/ɍ�`�������\�� �J�)c�4UA��/�U5�3���oOp���6iJnΓn�7�����|خ��F��t�����h��Q�@o�+�+�li���G�
oW�0Z� �$:���Pe�U���7%��VhQjV�(�Bˍ�Z�b�;qJ���z��Xj9uH�%%�=�����uzh֕}�xp�����
�dۉ��n�ն�v��u��-��o|�&��c�4͖�Ą��w�5P�P���& �a�v��cH�[>I(�1�#(�_'�1���J}�Bۑ?�����ʟ5����=ņ��������'��f� *���         �  x����n�@����l�9g.0�K�(J���R��T)�0uHl� �jީ��͋�X��m7^Y�?���x�޵��5�`�,!i�|���H������=��o/.��MU��WS���y  `��"�(b!|G��'!��.�&�e���[�_��>/Kr5O��l9�H�������߄���v��EV�l9�T�,̴��IX�W��������?��^�l����1���2K&3s� 5�aqA�/�?%D	�X����c���W�UQ��G
� �S��U�j�F��j��+ˋ�U#���1���'B�Ξg���5E�t���TAZ���Ե��Q���5I�L��ס��!��]��N`���liWz�\㶷��i�t����4H)��B���[��~�]��yTyc�`�/�I9�=$i��[�#��(Ę��-'U���=���u��)}&R��(�ɟ�E��`�+���J��,�$xʟ��C��SJSԨBs���`��V��I�˱�}��3� h��v��+��Mޑ��G���]3��U�U �vD`$���nף�qZ��ჯ�c��nuKh�ve���F'� ߮���U�-�I8`��?k���|���0�HA������F�SD������u�8:���Tidy��}�2zG� ����
��     