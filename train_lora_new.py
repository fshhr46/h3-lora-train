        print(f"🖥️  设备: {self.device}")

        # 使用本地缓存目录，避免网络锁
        model_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".models")
        # 兼容 Linux 和 macOS 路径
        if not os.path.isdir(model_cache):
            model_cache = os.path.expanduser("~/.cache/huggingface/hub")

        print(f"📦 Model cache: {model_cache}")

        # 手动加载各组件（避免 pipeline 触发完整下载）
        from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        print("Loading UNet (fp16)...")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_cache, subfolder="models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/unet",
            torch_dtype=torch.float16,
        )

        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            model_cache, subfolder="models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/vae",
            torch_dtype=torch.float32,
        ).eval()

        print("Loading text_encoder...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_cache, subfolder="models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/text_encoder",
            torch_dtype=torch.float16,
        ).eval()

        print("Loading text_encoder_2...")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_cache, subfolder="models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/text_encoder_2",
            torch_dtype=torch.float16,
        ).eval()

        print("Loading tokenizers...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_cache, subfolder="models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/tokenizer",
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_cache, subfolder="models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/tokenizer_2",
        )

        print("Loading noise scheduler...")
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            model_cache, subfolder="models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/scheduler",
        )

        # 冻结 components，只训练 LoRA on unet
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder_2.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

        # 转到 device
        self.text_encoder.to(self.device).eval()
        self.text_encoder_2.to(self.device).eval()
        self.vae.to(self.device).eval()
        self.unet.to(self.device).eval()

        # 配置 LoRA (在 fp16 unet 上)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",
                "query", "key", "value", "out",
                "conv1", "conv2",
                "conv_shortcut",
            ],
            lora_dropout=0.1,
        )

        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
